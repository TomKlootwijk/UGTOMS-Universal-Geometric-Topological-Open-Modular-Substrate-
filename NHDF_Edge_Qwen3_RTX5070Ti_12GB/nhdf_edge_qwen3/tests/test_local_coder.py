from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "local_coder.py"
CONFIG = ROOT / "configs" / "opencode_nhdf_local.json"


def _load_launcher() -> ModuleType:
    spec = importlib.util.spec_from_file_location("local_coder_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_is_local_only_and_has_large_context() -> None:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    model_id = "local-runtime/local-qwen3-30b-a3b"
    model = value["provider"]["local-runtime"]["models"]["local-qwen3-30b-a3b"]

    assert value["model"] == model_id
    assert value["small_model"] == model_id
    assert value["enabled_providers"] == ["local-runtime"]
    assert value["provider"]["local-runtime"]["options"]["baseURL"] == (
        "{env:UGTOMS_LOCAL_CODER_BASE_URL}/v1"
    )
    assert model["limit"] == {"context": 32768, "output": 4096}
    assert model["tool_call"] is True
    assert value["share"] == "disabled"
    assert value["autoupdate"] is False
    assert value["subagent_depth"] == 0
    assert value["compaction"]["reserved"] == 2048
    prompt = value["agent"]["local-coder"]["prompt"]
    assert prompt == _load_launcher().CANONICAL_AGENT_PROMPT
    assert "digest-verified UGTOMS contract" in prompt


def test_config_permissions_fail_closed() -> None:
    permission = json.loads(CONFIG.read_text(encoding="utf-8"))["permission"]

    assert permission["*"] == "deny"
    for allowed in ("read", "glob", "grep", "list", "lsp"):
        assert permission[allowed] == "allow"
    assert permission["edit"] == "ask"
    assert permission["bash"]["*"] == "ask"
    for denied in ("task", "external_directory", "webfetch", "websearch", "skill"):
        assert permission[denied] == "deny"
    for command in (
        "git commit*",
        "git push*",
        "git reset --hard*",
        "git clean*",
        "rm *",
        "Remove-Item *",
    ):
        assert permission["bash"][command] == "deny"


def test_config_validator_rejects_malformed_permission_shape() -> None:
    launcher = _load_launcher()
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    value["permission"] = "allow"

    with pytest.raises(launcher.LocalCoderError, match="permissions must be an object"):
        launcher._validate_config_contract(
            value,
            expected_base_url="{env:NHDF_LOCAL_CODER_BASE_URL}/v1",
        )


def test_isolated_environment_overrides_project_config_layers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    state = tmp_path / "state"
    monkeypatch.setattr(launcher, "LOCAL_STATE_ROOT", state)
    monkeypatch.setenv("OPENCODE_CONFIG", "inherited.json")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"share":"auto"}')
    monkeypatch.setenv("HTTPS_PROXY", "http://external-proxy.invalid")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")

    environment = launcher.isolated_environment(
        CONFIG, "http://127.0.0.1:19090"
    )
    inline = json.loads(environment["OPENCODE_CONFIG_CONTENT"])

    assert environment["OPENCODE_CONFIG"] == str(CONFIG)
    assert environment["OPENCODE_CONFIG_DIR"] == str(state / "config-dir")
    assert environment["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert environment["OPENCODE_DISABLE_DEFAULT_PLUGINS"] == "1"
    assert environment["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
    assert environment["OPENCODE_PURE"] == "1"
    assert "HTTPS_PROXY" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert environment["no_proxy"] == "127.0.0.1,localhost"
    assert inline["share"] == "disabled"
    assert inline["provider"]["local-runtime"]["options"]["baseURL"] == (
        "http://127.0.0.1:19090/v1"
    )
    contract = Path(inline["instructions"][0])
    assert contract == state / "xdg-config" / "opencode" / "AGENTS.md"
    assert contract.read_bytes() == launcher.SUBSTRATE_CONTRACT.read_bytes()


def test_parse_noninteractive_arguments() -> None:
    launcher = _load_launcher()

    options = launcher.parse_options(
        ["C:/work/repo with spaces", "--quick", "--run", "--format", "json", "fix tests"]
    )

    assert options.target == Path("C:/work/repo with spaces")
    assert options.quick is True
    assert options.run_args == ("--format", "json", "fix tests")


@pytest.mark.parametrize(
    "unsafe",
    [
        "--share",
        "--auto",
        "--model=cloud/model",
        "--agent",
        "--attach",
        "-f",
        "-mcloud/model",
    ],
)
def test_parse_rejects_run_options_that_bypass_isolation(unsafe: str) -> None:
    launcher = _load_launcher()

    with pytest.raises(launcher.LocalCoderError, match="isolation contract"):
        launcher.parse_options(["C:/work/repo", "--run", unsafe, "value"])


def test_validate_git_target_accepts_worktree_subdirectory(tmp_path: Path) -> None:
    launcher = _load_launcher()
    repository = tmp_path / "repo with spaces"
    nested = repository / "src"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)

    target, root = launcher.validate_git_target(nested)

    assert target == nested.resolve()
    assert root == repository.resolve()


def test_validate_git_target_rejects_plain_directory(tmp_path: Path) -> None:
    launcher = _load_launcher()

    with pytest.raises(launcher.LocalCoderError, match="Git working tree"):
        launcher.validate_git_target(tmp_path)


@pytest.mark.skipif(
    not (ROOT / ".local-coder/node_modules/opencode-ai/bin/opencode.exe").is_file(),
    reason="project-local OpenCode has not been installed",
)
def test_pinned_client_resolves_only_the_isolated_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    repository = tmp_path / "untrusted-project-config"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / "opencode.json").write_text(
        json.dumps(
            {
                "model": "openai/cloud-model",
                "enabled_providers": ["openai"],
                "share": "auto",
                "permission": "allow",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "LOCAL_STATE_ROOT", tmp_path / "isolated-state")
    environment = launcher.isolated_environment(
        CONFIG, "http://127.0.0.1:19090"
    )

    launcher.validate_resolved_config(
        launcher.OPENCODE_EXE,
        repository,
        environment,
        expected_base_url="http://127.0.0.1:19090",
        expected_contract_path=(
            Path(environment["XDG_CONFIG_HOME"]) / "opencode" / "AGENTS.md"
        ),
    )


def test_owned_server_is_stopped_and_quick_mode_is_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    target = tmp_path / "target"
    artifact = tmp_path / "artifact"
    executable = tmp_path / "opencode.exe"
    target.mkdir()
    artifact.mkdir()
    executable.write_bytes(b"stub")
    observed: dict[str, object] = {}

    class FakeServer:
        base_url = "http://127.0.0.1:19090"

        def __init__(self, passed_artifact: Path, **kwargs: object) -> None:
            observed["server_init"] = (passed_artifact, kwargs)

        def start(self) -> None:
            observed["started"] = True

        def stop(self) -> None:
            observed["stopped"] = True

    def fake_client(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["client"] = (command, kwargs)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(launcher, "OPENCODE_EXE", executable)
    monkeypatch.setattr(launcher, "validate_git_target", lambda path: (target, target))
    monkeypatch.setattr(launcher, "validate_local_install", lambda _path: None)
    monkeypatch.setattr(launcher, "validate_agent_artifact", lambda _path: {})
    monkeypatch.setattr(launcher, "isolated_environment", lambda *_args: {"isolated": "yes"})
    monkeypatch.setattr(launcher, "validate_resolved_config", lambda *_args, **_kwargs: None)
    options = launcher.LaunchOptions(
        target=target,
        artifact=artifact,
        config=CONFIG,
        port=19090,
        threads=4,
        quick=True,
        startup_timeout=30.0,
        run_args=("--format", "json", "inspect"),
    )

    exit_code = launcher.run_local_coder(
        options, server_factory=FakeServer, client_runner=fake_client
    )

    assert exit_code == 7
    assert observed["started"] is True
    assert observed["stopped"] is True
    _, server_kwargs = observed["server_init"]
    assert server_kwargs["verify_payload_hash"] is False
    command, client_kwargs = observed["client"]
    assert command == [
        str(executable),
        "--pure",
        "run",
        "--model",
        launcher.MODEL_ID,
        "--agent",
        "local-coder",
        "--format",
        "json",
        "inspect",
    ]
    assert client_kwargs["cwd"] == str(target)
    assert client_kwargs["env"] == {"isolated": "yes"}


def test_owned_server_is_stopped_when_client_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    target = tmp_path / "target"
    artifact = tmp_path / "artifact"
    executable = tmp_path / "opencode.exe"
    target.mkdir()
    artifact.mkdir()
    executable.write_bytes(b"stub")
    stopped = False

    class FakeServer:
        base_url = "http://127.0.0.1:18080"

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            nonlocal stopped
            stopped = True

    def fail_client(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("client failed")

    monkeypatch.setattr(launcher, "OPENCODE_EXE", executable)
    monkeypatch.setattr(launcher, "validate_git_target", lambda path: (target, target))
    monkeypatch.setattr(launcher, "validate_local_install", lambda _path: None)
    monkeypatch.setattr(launcher, "validate_agent_artifact", lambda _path: {})
    monkeypatch.setattr(launcher, "validate_resolved_config", lambda *_args, **_kwargs: None)
    options = launcher.LaunchOptions(
        target=target,
        artifact=artifact,
        config=CONFIG,
        port=18080,
        threads=None,
        quick=False,
        startup_timeout=30.0,
        run_args=None,
    )

    with pytest.raises(OSError, match="client failed"):
        launcher.run_local_coder(
            options, server_factory=FakeServer, client_runner=fail_client
        )

    assert stopped is True


def test_setup_script_pins_a_project_local_install() -> None:
    script = (ROOT / "scripts" / "setup_local_coder.ps1").read_text(encoding="utf-8")

    assert 'PinnedVersion = "1.18.25"' in script
    assert "--prefix $InstallRoot" in script
    assert "--cache $NpmCache" in script
    assert "--save-exact" in script
    assert " -g " not in script
