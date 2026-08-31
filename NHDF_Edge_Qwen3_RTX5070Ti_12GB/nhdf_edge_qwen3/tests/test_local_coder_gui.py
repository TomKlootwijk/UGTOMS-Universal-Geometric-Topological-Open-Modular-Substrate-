from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUI_PATH = PROJECT_ROOT / "scripts" / "local_coder_gui.py"


def _load_gui():
    name = "local_coder_gui_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, GUI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def gui():
    return _load_gui()


def _control_record(payload: bytes) -> dict[str, object]:
    revision = "6c6e8692f43e4ca663f7ece8229a1361090d3a4c"
    repository = "owner/repository"
    artifact = "tiny.gguf"
    return {
        "role": "test",
        "backend": "GGUF / llama.cpp",
        "artifact": artifact,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_repository": repository,
        "source_revision": revision,
        "immutable_url": (
            f"https://huggingface.co/{repository}/resolve/{revision}/{artifact}"
            "?download=true"
        ),
    }


def _write_control(root: Path, payload: bytes) -> Path:
    directory = root / "model"
    directory.mkdir(parents=True)
    path = directory / "CONTROL_SOURCE.json"
    path.write_text(json.dumps(_control_record(payload)), encoding="utf-8")
    return path


def _load_control(gui, root: Path, payload: bytes):
    path = _write_control(root, payload)
    raw = path.read_bytes()
    return gui.load_model_control(
        path,
        trusted_root=root,
        expected_source_bytes=len(raw),
        expected_source_sha256=hashlib.sha256(raw).hexdigest(),
    )


class _FakeResponse:
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        headers: dict[str, str] | None = None,
        url: str = "https://cdn.example.invalid/model",
    ) -> None:
        self.payload = payload
        self.status = status
        self.headers = headers or {}
        self.url = url
        self.offset = 0
        self.closed = False

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self.payload) - self.offset
        block = self.payload[self.offset : self.offset + amount]
        self.offset += len(block)
        return block

    def geturl(self) -> str:
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True


class _FakeOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests = []

    def open(self, request, timeout: float):
        self.requests.append((request, timeout))
        return self.response


def test_control_source_rejects_duplicate_keys_and_nonfinite_json(
    gui, tmp_path: Path
) -> None:
    control = _write_control(tmp_path, b"model")
    raw = control.read_text(encoding="utf-8")
    duplicate = raw[:-1] + ',"bytes":5}'
    control.write_text(duplicate, encoding="utf-8")

    with pytest.raises(gui.GuiError, match="duplicate JSON key"):
        gui.load_model_control(control, trusted_root=tmp_path)

    control.write_text(raw.replace('"bytes": 5', '"bytes": NaN'), encoding="utf-8")
    with pytest.raises(gui.GuiError, match="non-finite JSON"):
        gui.load_model_control(control, trusted_root=tmp_path)


def test_control_source_requires_exact_immutable_https_revision(
    gui, tmp_path: Path
) -> None:
    control = _write_control(tmp_path, b"model")
    value = json.loads(control.read_text(encoding="utf-8"))
    value["immutable_url"] = "http://huggingface.co/owner/repository/tiny.gguf"
    control.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(gui.GuiError, match="immutable HTTPS"):
        gui.load_model_control(control, trusted_root=tmp_path)


def test_downloader_resumes_exact_range_and_promotes_atomically(
    gui, tmp_path: Path
) -> None:
    payload = b"abcdefghijklmno"
    control = _load_control(gui, tmp_path, payload)
    partial = control.destination.with_name(control.destination.name + ".download.part")
    partial.write_bytes(payload[:5])
    response = _FakeResponse(
        payload[5:],
        status=206,
        headers={"Content-Range": f"bytes 5-{len(payload) - 1}/{len(payload)}"},
    )
    opener = _FakeOpener(response)
    progress = []
    downloader = gui.ModelDownloader(
        control,
        trusted_root=tmp_path,
        opener=opener,
        chunk_bytes=3,
    )

    result = downloader.download(on_progress=progress.append)

    assert result == control.destination
    assert result.read_bytes() == payload
    assert not partial.exists()
    request, _timeout = opener.requests[0]
    assert request.get_header("Range") == "bytes=5-"
    assert any(item.resumed for item in progress)


def test_downloader_restarts_safely_when_server_ignores_range(
    gui, tmp_path: Path
) -> None:
    payload = b"complete-payload"
    control = _load_control(gui, tmp_path, payload)
    partial = control.destination.with_name(control.destination.name + ".download.part")
    partial.write_bytes(b"wrong")
    opener = _FakeOpener(_FakeResponse(payload, status=200))

    result = gui.ModelDownloader(
        control,
        trusted_root=tmp_path,
        opener=opener,
        chunk_bytes=4,
    ).download()

    assert result.read_bytes() == payload


def test_downloader_cancel_keeps_partial_and_never_promotes(
    gui, tmp_path: Path
) -> None:
    payload = b"0123456789"
    control = _load_control(gui, tmp_path, payload)
    cancel = threading.Event()
    opener = _FakeOpener(_FakeResponse(payload, status=200))

    def progress(value) -> None:
        if value.phase == "downloading" and value.completed_bytes >= 3:
            cancel.set()

    with pytest.raises(gui.DownloadCancelled, match="kept for resume"):
        gui.ModelDownloader(
            control,
            trusted_root=tmp_path,
            opener=opener,
            chunk_bytes=3,
        ).download(cancel_event=cancel, on_progress=progress)

    partial = control.destination.with_name(control.destination.name + ".download.part")
    assert partial.read_bytes() == payload[:3]
    assert not control.destination.exists()


def test_downloader_rejects_wrong_range_and_does_not_promote(
    gui, tmp_path: Path
) -> None:
    payload = b"0123456789"
    control = _load_control(gui, tmp_path, payload)
    partial = control.destination.with_name(control.destination.name + ".download.part")
    partial.write_bytes(payload[:3])
    opener = _FakeOpener(
        _FakeResponse(
            payload[4:],
            status=206,
            headers={"Content-Range": f"bytes 4-9/{len(payload)}"},
        )
    )

    with pytest.raises(gui.GuiError, match="exact requested byte range"):
        gui.ModelDownloader(control, trusted_root=tmp_path, opener=opener).download()

    assert partial.read_bytes() == payload[:3]
    assert not control.destination.exists()


def test_downloader_rejects_symlink_partial(gui, tmp_path: Path) -> None:
    payload = b"0123456789"
    control = _load_control(gui, tmp_path, payload)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-model"
    outside.write_bytes(b"do not touch")
    partial = control.destination.with_name(control.destination.name + ".download.part")
    try:
        partial.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")

    with pytest.raises(gui.GuiError, match="symlink|reparse"):
        gui.ModelDownloader(
            control,
            trusted_root=tmp_path,
            opener=_FakeOpener(_FakeResponse(payload, status=200)),
        ).download()

    assert outside.read_bytes() == b"do not touch"


def test_parser_renders_text_tools_and_session_ids(gui) -> None:
    parser = gui.OpenCodeEventParser()
    text_event = parser.parse_line(
        json.dumps(
            {
                "type": "text",
                "sessionID": "ses_example_1",
                "part": {"text": "Finished the review."},
            }
        )
    )
    tool_event = parser.parse_line(
        json.dumps(
            {
                "type": "tool_use",
                "sessionID": "ses_example_1",
                "part": {
                    "tool": "read",
                    "state": {
                        "status": "completed",
                        "input": {"filePath": "README.md"},
                        "output": "hello",
                    },
                },
            }
        )
    )

    assert text_event.kind == "assistant"
    assert text_event.text == "Finished the review."
    assert text_event.session_id == "ses_example_1"
    assert tool_event.kind == "tool"
    assert "read" in tool_event.title
    assert "README.md" in tool_event.text


def test_parser_rejects_ambiguous_or_duplicate_session_records(gui) -> None:
    parser = gui.OpenCodeEventParser()
    conflicting = parser.parse_line(
        json.dumps(
            {
                "type": "text",
                "sessionID": "ses_one",
                "part": {"sessionID": "ses_two", "text": "unsafe"},
            }
        )
    )
    duplicate = parser.parse_line(
        '{"type":"text","type":"tool_use","part":{"text":"x"}}'
    )

    assert conflicting.kind == "error"
    assert conflicting.session_id is None
    assert duplicate.kind == "diagnostic"
    assert "duplicate JSON key" in duplicate.detail


def test_readiness_snapshot_reports_all_independent_cards(gui) -> None:
    def missing() -> str:
        raise FileNotFoundError("download it")

    def blocked() -> str:
        raise RuntimeError("GPU is busy")

    probe = gui.ReadinessProbe(
        {
            "client": lambda: "client ready",
            "model": missing,
            "runtime": lambda: "runtime ready",
            "gpu": blocked,
            "artifact": lambda: "artifact ready",
        }
    )

    result = probe.probe()

    assert len(result.cards) == 5
    assert result.by_key("client").state is gui.ReadinessState.READY
    assert result.by_key("model").state is gui.ReadinessState.MISSING
    assert result.by_key("gpu").state is gui.ReadinessState.BLOCKED
    assert result.ready_to_start is False


def test_runtime_readiness_includes_exact_cuda_dependency_hashes(
    gui, tmp_path: Path
) -> None:
    runtime = tmp_path / "runtime.dll"
    runtime.write_bytes(b"runtime")
    calls = []

    def cuda_preflight():
        calls.append(True)
        return SimpleNamespace(
            version="12.8",
            verified_dependency_count=3,
            dependency_names=("cudart.dll", "cublas.dll", "cublasLt.dll"),
        )

    launcher = SimpleNamespace(
        PROJECT_ROOT=tmp_path,
        CANONICAL_RUNTIME_PATHS=("runtime.dll",),
        CANONICAL_REFERENCE_RECORDS={
            "runtime.dll": (7, hashlib.sha256(b"runtime").hexdigest())
        },
        hybrid=SimpleNamespace(
            preflight_windows_cuda_dependencies=cuda_preflight,
        ),
    )

    detail = gui.verify_runtime_readiness(launcher, include_cuda=True)
    assert "CUDA 12.8 (3 DLLs)" in detail
    assert calls == [True]

    def fail_cuda():
        raise OSError("CUDA dependency digest mismatch")

    launcher.hybrid.preflight_windows_cuda_dependencies = fail_cuda
    with pytest.raises(OSError, match="CUDA dependency digest mismatch"):
        gui.verify_runtime_readiness(launcher, include_cuda=True)


def test_windows_powershell_resolution_uses_exact_system32_subdirectory(
    gui, tmp_path: Path
) -> None:
    system32 = tmp_path / "Windows" / "System32"
    expected = system32 / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    observed = {}

    def require(path: Path, **kwargs: object) -> Path:
        observed["path"] = path
        observed.update(kwargs)
        return path

    launcher = SimpleNamespace(
        hybrid=SimpleNamespace(_windows_system_directory=lambda: system32),
        _require_unaliased_executable=require,
    )

    assert gui.resolve_windows_powershell(launcher) == expected
    assert observed["path"] == expected
    assert observed["trusted_root"] == Path(expected.anchor)


def test_setup_cancel_terminates_owned_process_tree(
    gui, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Process:
        def __init__(self) -> None:
            self.returncode = None
            self.waited = False

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            self.waited = True
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    terminated = []

    def terminate(owned) -> None:
        terminated.append(owned)
        owned.returncode = -1

    monkeypatch.setattr(
        gui.SubprocessClientExecutor,
        "_terminate_process_tree",
        staticmethod(terminate),
    )
    runner = gui.VerifiedSetupRunner.__new__(gui.VerifiedSetupRunner)
    runner._lock = threading.RLock()
    runner._process = process

    runner.cancel()

    assert terminated == [process]
    assert process.waited is True


def test_root_entrypoint_resolves_self_and_prefers_python_before_py() -> None:
    launcher = (PROJECT_ROOT / "START_LOCAL_CODER.cmd").read_text(encoding="utf-8")

    assert 'set "APP_ROOT=%~dp0"' in launcher
    assert 'set "GUI_SCRIPT=%APP_ROOT%scripts\\local_coder_gui.py"' in launcher
    assert launcher.index('where.exe" python.exe') < launcher.index('where.exe" py.exe')
    assert 'start "" /B /D "%APP_ROOT%"' in launcher
    assert "Python 3 with tkinter" in launcher


class _FakeLauncher:
    DEFAULT_PORT = 18080

    def __init__(self, root: Path) -> None:
        self.root = root
        self.DEFAULT_CONFIG = root / "config.json"
        self.OPENCODE_EXE = root / "opencode.exe"
        self.DEFAULT_ARTIFACT = root / "artifact"
        self.calls = []

    def validate_git_target(self, target: Path):
        target = target.resolve()
        self.calls.append("target")
        return target, target

    def validate_config(self, path: Path):
        self.calls.append("config")
        return SimpleNamespace(path=path, raw=b"{}", sha256="a" * 64)

    def validate_local_install(self, path: Path):
        self.calls.append("client")
        return path

    def resolve_canonical_artifact(self, path: Path):
        return path

    def validate_agent_artifact(self, path: Path):
        self.calls.append("artifact")
        return SimpleNamespace(server_approval="approved")

    def isolated_environment(self, _config, base_url: str):
        return {
            "XDG_CONFIG_HOME": str(self.root / "xdg"),
            "UGTOMS_LOCAL_CODER_BASE_URL": base_url,
        }

    def verify_execution_inputs(self, *_args: object):
        self.calls.append("verify")

    def validate_resolved_config(self, *_args: object, **_kwargs: object):
        self.calls.append("resolved")

    def build_opencode_command(self, executable: Path, run_args: tuple[str, ...]):
        return [str(executable), "--pure", "run", *run_args]


class _FakeServer:
    def __init__(self, *, health_failures: int = 0) -> None:
        self.base_url = "http://127.0.0.1:18080"
        self.is_running = False
        self.health_failures = health_failures
        self.start_wait_ready = None
        self.stopped = False

    def start(self, *, wait_ready: bool = True):
        self.start_wait_ready = wait_ready
        self.is_running = True
        return self

    def health(self, *, timeout_seconds: float = 2.0):
        del timeout_seconds
        if self.health_failures:
            self.health_failures -= 1
            from nhdf_edge.server import HybridServerUnavailableError

            raise HybridServerUnavailableError("still loading")
        if not self.is_running:
            raise RuntimeError("stopped")
        return {"status": "ok"}

    def stop(self):
        self.is_running = False
        self.stopped = True


class _FakeSetup:
    def cancel(self) -> None:
        pass

    def run(self, **_kwargs: object):
        return Path("opencode.exe")


class _FakeDownloader:
    def download(self, **_kwargs: object):
        return Path("model.gguf")


class _FakeExecutor:
    def __init__(self) -> None:
        self.commands = []
        self.next_session = "ses_gui_owned"

    def cancel(self) -> None:
        pass

    def run(self, command, **_kwargs):
        self.commands.append(list(command))
        return SimpleNamespace(returncode=0, session_id=self.next_session)


def _controller(gui, tmp_path: Path, server: _FakeServer, executor=None):
    launcher = _FakeLauncher(tmp_path)
    probe = gui.ReadinessProbe(
        {key: (lambda key=key: key) for key, _label in gui.ReadinessProbe.CARD_ORDER}
    )
    controller = gui.LocalCoderController(
        launcher=launcher,
        server_factory=lambda *_args, **_kwargs: server,
        readiness_probe=probe,
        downloader=_FakeDownloader(),
        setup_runner=_FakeSetup(),
        client_executor=executor or _FakeExecutor(),
        startup_timeout_seconds=0.25,
        startup_poll_seconds=0.001,
    )
    return controller, launcher


def test_start_uses_nonblocking_server_start_and_cancel_stops_process(
    gui, tmp_path: Path
) -> None:
    repository = tmp_path.resolve()
    server = _FakeServer(health_failures=1000)
    controller, _launcher = _controller(gui, tmp_path, server)
    cancel = threading.Event()

    def status(message: str) -> None:
        if "loading" in message.lower():
            cancel.set()

    with pytest.raises(gui.OperationCancelled, match="process was stopped"):
        controller.start(repository, cancel_event=cancel, on_status=status)

    assert server.start_wait_ready is False
    assert server.stopped is True
    assert controller.is_running is False


def test_controller_keeps_server_resident_and_binds_owned_session(
    gui, tmp_path: Path
) -> None:
    server = _FakeServer()
    executor = _FakeExecutor()
    controller, _launcher = _controller(gui, tmp_path, server, executor)
    controller.start(tmp_path)

    controller.send_prompt("inspect it", mode=gui.InteractionMode.READ_ONLY)
    assert "--auto" not in executor.commands[0]
    assert "--session" not in executor.commands[0]
    assert controller.session_id == "ses_gui_owned"
    assert server.is_running is True

    with pytest.raises(gui.GuiError, match="explicit confirmation"):
        controller.send_prompt("fix it", mode=gui.InteractionMode.SCOPED_EDITS)
    controller.authorize_scoped_work(confirmed=True)
    controller.send_prompt("fix it", mode=gui.InteractionMode.SCOPED_EDITS)
    assert "--auto" in executor.commands[1]
    session_index = executor.commands[1].index("--session")
    assert executor.commands[1][session_index + 1] == "ses_gui_owned"

    controller.new_session()
    assert controller.session_id is None
    assert controller.scoped_work_authorized is False
    controller.stop()
    assert server.stopped is True


def test_executor_callback_failure_terminates_owned_process_and_drains_pipes(
    gui, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Pipe:
        def __init__(self, lines):
            self.lines = list(lines)
            self.closed = False

        def __iter__(self):
            return iter(self.lines)

        def close(self):
            self.closed = True

    class Process:
        def __init__(self):
            self.stdout = Pipe(['{"type":"text","part":{"text":"hello"}}\n'])
            self.stderr = Pipe([])
            self.returncode = None
            self.pid = 123

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            del timeout
            if self.returncode is None:
                self.returncode = -9
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Process()
    monkeypatch.setattr(gui.subprocess, "Popen", lambda *_args, **_kwargs: process)
    terminated = []

    def terminate(owned) -> None:
        terminated.append(owned)
        owned.returncode = -9

    monkeypatch.setattr(
        gui.SubprocessClientExecutor,
        "_terminate_process_tree",
        staticmethod(terminate),
    )
    executor = gui.SubprocessClientExecutor()

    with pytest.raises(RuntimeError, match="display callback failed"):
        executor.run(
            ["opencode.exe"],
            cwd=PROJECT_ROOT,
            environment={},
            cancel_event=threading.Event(),
            on_event=lambda _event: (_ for _ in ()).throw(
                RuntimeError("display callback failed")
            ),
        )

    assert terminated == [process]
    assert process.stdout.closed is True
    assert process.stderr.closed is True


def test_windows_cancellation_uses_absolute_taskkill_tree_switches(
    gui, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nhdf_edge import hybrid

    taskkill = tmp_path / "taskkill.exe"
    taskkill.write_bytes(b"stub")
    monkeypatch.setattr(gui.os, "name", "nt")
    monkeypatch.setattr(hybrid, "_trusted_system_executable", lambda _name: taskkill)
    monkeypatch.setattr(
        hybrid,
        "_minimal_subprocess_environment",
        lambda **_kwargs: {"PATH": str(tmp_path)},
    )
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(gui.subprocess, "run", run)
    process = SimpleNamespace(pid=987, poll=lambda: None, kill=lambda: None)

    gui.SubprocessClientExecutor._terminate_process_tree(process)

    assert observed["command"] == [str(taskkill), "/PID", "987", "/T", "/F"]
    assert Path(observed["command"][0]).is_absolute()
    assert observed["kwargs"]["shell"] is False
