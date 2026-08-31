from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import zipfile
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


def _powershell_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _run_bundled_archive_installer(
    project: Path,
    archive: Path,
    executable_bytes: bytes,
    *,
    archive_sha256: str | None = None,
) -> subprocess.CompletedProcess[str]:
    executable_sha256 = hashlib.sha256(executable_bytes).hexdigest()
    expected_archive_sha256 = archive_sha256 or hashlib.sha256(archive.read_bytes()).hexdigest()
    install_root = project / ".local-coder"
    executable = install_root / "node_modules/opencode-ai/bin/opencode.exe"
    script = ROOT / "scripts" / "setup_local_coder.ps1"
    command = "; ".join(
        (
            f". {_powershell_literal(script)} -FunctionsOnly",
            f"$ProjectRoot = {_powershell_literal(project)}",
            f"$InstallRoot = {_powershell_literal(install_root)}",
            f"$NpmCache = {_powershell_literal(install_root / 'npm-cache')}",
            f"$OpenCodeExe = {_powershell_literal(executable)}",
            f"$BundledArchive = {_powershell_literal(archive)}",
            f"$PinnedArchiveBytes = {archive.stat().st_size}",
            f"$PinnedArchiveSha256 = '{expected_archive_sha256}'",
            f"$PinnedExecutableBytes = {len(executable_bytes)}",
            f"$PinnedSha256 = '{executable_sha256}'",
            "Install-OpenCodeFromBundledArchive",
        )
    )
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _validated_canonical_manifest(launcher: ModuleType) -> dict:
    manifest = copy.deepcopy(
        launcher.hybrid.load_hybrid_manifest(launcher.DEFAULT_ARTIFACT)
    )
    evidence = launcher.DEFAULT_ARTIFACT / "evidence" / "functional_gate.json"
    manifest["validation"].update(
        {
            "status": "VALIDATED",
            "deployment_loadable": True,
            "evidence": {
                "path": "evidence/functional_gate.json",
                "bytes": evidence.stat().st_size,
                "sha256": launcher._sha256_file(evidence),
            },
        }
    )
    records = {
        launcher.CANONICAL_PAYLOAD_PATH: manifest["payload"],
        launcher.CANONICAL_SOURCE_RECORD_PATH: manifest["source_record"],
        launcher.CANONICAL_SPECIFICATION_PATH: manifest["substrate"]["specification"],
        **{
            path: record
            for path, record in zip(
                launcher.CANONICAL_RUNTIME_PATHS, manifest["runtime"]["files"]
            )
        },
        **{
            path: record
            for path, record in zip(
                launcher.CANONICAL_ASSURANCE_PATHS,
                manifest["assurance_evidence"],
            )
        },
        launcher.CANONICAL_VALIDATION_EVIDENCE_PATH: manifest["validation"]["evidence"],
    }
    for path, record in records.items():
        expected_bytes, expected_sha256 = launcher.CANONICAL_REFERENCE_RECORDS[path]
        record.update(bytes=expected_bytes, sha256=expected_sha256)
    return manifest


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


def test_agent_contract_names_graph_v2_roles_evidence_and_feedback_boundaries() -> None:
    launcher = _load_launcher()
    contract = launcher.SUBSTRATE_CONTRACT.read_text(encoding="utf-8")

    for required in (
        "graph-v2",
        "dependency ID-to-hash",
        "definition_ref",
        "definition_hash",
        "b_payload",
        "b_topology",
        "b_jitter",
        "b_branch",
        "separately recorded, typed feedback",
        "unrelated to mathematical fixed-point iteration",
        "Selecting a profile is an evidence-bearing act",
        "automatic promotion remains prohibited",
    ):
        assert required in contract


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
    config = launcher.validate_config(CONFIG)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "LOCAL_STATE_ROOT", state)
    monkeypatch.setenv("OPENCODE_CONFIG", "inherited.json")
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", '{"share":"auto"}')
    monkeypatch.setenv("HTTPS_PROXY", "http://external-proxy.invalid")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-leak")

    environment = launcher.isolated_environment(
        config, "http://127.0.0.1:19090"
    )
    inline = json.loads(environment["OPENCODE_CONFIG_CONTENT"])

    assert environment["OPENCODE_CONFIG"] == str(state / "config-dir" / "opencode.json")
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
        "--no-auto",
        "--model=cloud/model",
        "--agent",
        "--attach",
        "--command",
        "--continue",
        "--hostname=0.0.0.0",
        "--cors=*",
        "--pure",
        "--no-pure",
        "--pure=false",
        "--no-pure=true",
        "-c",
        "-f",
        "-ic",
        "-imcloud/model",
        "-mcloud/model",
        "-sforeign-session",
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


def test_git_resolver_ignores_hostile_target_and_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    fake = tmp_path / ("git.exe" if os.name == "nt" else "git")
    fake.write_bytes(b"hostile target-local executable")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))

    resolved = launcher._resolve_git_executable()

    assert resolved.is_absolute()
    assert resolved != fake.resolve()


def test_git_target_probe_uses_exact_executable_safe_cwd_and_scrubbed_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    repository = tmp_path / "repo"
    target = repository / "src"
    trusted_git = tmp_path / "machine-git" / "git.exe"
    target.mkdir(parents=True)
    trusted_git.parent.mkdir()
    trusted_git.write_bytes(b"fixture")
    hostile = target / "git.exe"
    hostile.write_bytes(b"hostile")
    monkeypatch.setenv("PATH", str(target))
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-leak")
    monkeypatch.setattr(launcher, "_resolve_git_executable", lambda: trusted_git.resolve())
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout=f"{repository.resolve()}\n", stderr=""
        )

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    resolved_target, worktree = launcher.validate_git_target(target)

    assert resolved_target == target.resolve()
    assert worktree == repository.resolve()
    command = observed["command"]
    kwargs = observed["kwargs"]
    assert command[0] == str(trusted_git.resolve())
    assert command[1:3] == ["-C", str(target.resolve())]
    assert kwargs["cwd"] == str(trusted_git.resolve().parent)
    assert kwargs["env"]["NoDefaultCurrentDirectoryInExePath"] == "1"
    assert str(target) not in kwargs["env"].get("PATH", "")
    assert "GITHUB_TOKEN" not in kwargs["env"]


def test_gpu_probes_use_exact_system_executable_and_scrubbed_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    system_directory = tmp_path / "system"
    hostile_directory = tmp_path / "target"
    system_directory.mkdir()
    hostile_directory.mkdir()
    trusted = system_directory / "nvidia-smi.exe"
    trusted.write_bytes(b"fixture")
    (hostile_directory / "nvidia-smi.exe").write_bytes(b"hostile")
    monkeypatch.setenv("PATH", str(hostile_directory))
    monkeypatch.setenv("NVIDIA_API_KEY", "must-not-leak")
    monkeypatch.setattr(
        launcher.hybrid, "_trusted_system_executable", lambda _name: trusted.resolve()
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "NVIDIA GeForce RTX 5070 Ti Laptop GPU\n"
        if "memory.total" in command[1]:
            stdout = "12227, 1000, 12\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(launcher.hybrid.subprocess, "run", fake_run)

    assert launcher.hybrid._gpu_name() == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
    assert launcher.hybrid._gpu_sample() == (12227, 1000, 12)
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[0] == str(trusted.resolve())
        assert str(hostile_directory) not in kwargs["env"].get("PATH", "")
        assert "NVIDIA_API_KEY" not in kwargs["env"]


def test_canonical_artifact_path_rejects_an_alternate_self_signed_pack(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    alternate = tmp_path / "self-signed-artifact"
    alternate.mkdir()
    (alternate / launcher.hybrid.HYBRID_MANIFEST).write_text(
        json.dumps({"format": launcher.hybrid.HYBRID_FORMAT}),
        encoding="utf-8",
    )

    with pytest.raises(launcher.LocalCoderError, match="only the canonical sealed"):
        launcher.resolve_canonical_artifact(alternate)

    alias = launcher.DEFAULT_ARTIFACT / ".." / launcher.DEFAULT_ARTIFACT.name
    assert launcher.resolve_canonical_artifact(alias) == launcher.DEFAULT_ARTIFACT.resolve()


def test_artifact_reference_resolver_rejects_relative_and_absolute_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    project = tmp_path / "project"
    artifact = project / "packs" / "canonical"
    artifact.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)

    with pytest.raises(launcher.LocalCoderError, match="outside the intended project"):
        launcher._resolve_artifact_reference(
            artifact, "../../../outside.json", "escaped evidence"
        )
    with pytest.raises(launcher.LocalCoderError, match="repository-relative"):
        launcher._resolve_artifact_reference(
            artifact, str(outside.resolve()), "absolute evidence"
        )


def test_canonical_manifest_references_and_sealed_records_are_accepted() -> None:
    launcher = _load_launcher()
    manifest = _validated_canonical_manifest(launcher)

    launcher._validate_artifact_references(
        launcher.DEFAULT_ARTIFACT.resolve(), manifest
    )


def test_rewritten_evidence_and_self_resealed_record_cannot_replace_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    project = tmp_path / "project"
    artifact = project / "packs" / "qwen3-30b-a3b-iq2m-32k-q4kv"
    evidence = artifact / "evidence" / "functional_gate.json"
    snapshot = project / "metrics" / "local" / "ugtoms_local_agent_32k" / "functional_gate.json"
    evidence.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    trusted = b'{"passed":true}\n'
    rewritten = b'{"passed":true,"fabricated":true}\n'
    snapshot.write_bytes(trusted)
    evidence.write_bytes(rewritten)
    trusted_digest = hashlib.sha256(trusted).hexdigest()
    rewritten_digest = hashlib.sha256(rewritten).hexdigest()
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)
    monkeypatch.setattr(launcher, "EXPECTED_VALIDATION_EVIDENCE_BYTES", len(trusted))
    monkeypatch.setattr(launcher, "EXPECTED_VALIDATION_EVIDENCE_SHA256", trusted_digest)
    monkeypatch.setitem(
        launcher.CANONICAL_REFERENCE_RECORDS,
        launcher.CANONICAL_VALIDATION_EVIDENCE_PATH,
        (len(trusted), trusted_digest),
    )
    self_resealed = {
        "path": "evidence/functional_gate.json",
        "bytes": len(rewritten),
        "sha256": rewritten_digest,
    }

    with pytest.raises(launcher.LocalCoderError, match="byte contract changed|canonical SHA-256"):
        launcher._validate_evidence_snapshot(artifact, self_resealed)


def test_local_state_reparse_point_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    link = project / ".local-coder"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this Windows host")
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)

    with pytest.raises(launcher.LocalCoderError, match="symlink/reparse"):
        launcher._ensure_safe_project_directory(link / "state")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["payload"].update(path="../../README.md"),
            "model payload must reference exactly",
        ),
        (
            lambda value: value["payload"].update(sha256="0" * 64),
            "canonical SHA-256",
        ),
        (
            lambda value: value["runtime"].update(
                entrypoint=value["runtime"]["benchmark_entrypoint"]
            ),
            "runtime entrypoint must reference exactly",
        ),
        (
            lambda value: value["runtime"]["files"].reverse(),
            "runtime files:0 must reference exactly",
        ),
        (
            lambda value: value["assurance_evidence"].pop(),
            "exactly 3 canonical records",
        ),
        (
            lambda value: value["validation"]["evidence"].update(
                path="../../../../outside.json"
            ),
            "does not resolve|outside the intended project",
        ),
    ),
)
def test_artifact_reference_mutations_fail_closed(mutation, message: str) -> None:
    launcher = _load_launcher()
    manifest = _validated_canonical_manifest(launcher)
    mutation(manifest)

    with pytest.raises(launcher.LocalCoderError, match=message):
        launcher._validate_artifact_references(
            launcher.DEFAULT_ARTIFACT.resolve(), manifest
        )


def test_validate_agent_artifact_rejects_identity_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    manifest = _validated_canonical_manifest(launcher)
    manifest["model"]["source_revision"] = "0" * 40
    monkeypatch.setattr(
        launcher.hybrid,
        "load_hybrid_manifest_snapshot",
        lambda _artifact: (copy.deepcopy(manifest), "1" * 64),
    )

    with pytest.raises(launcher.LocalCoderError, match="model identity"):
        launcher.validate_agent_artifact(launcher.DEFAULT_ARTIFACT)


def test_validate_agent_artifact_accepts_only_the_canonical_bound_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_launcher()
    manifest = _validated_canonical_manifest(launcher)
    monkeypatch.setattr(
        launcher.hybrid,
        "load_hybrid_manifest_snapshot",
        lambda _artifact: (copy.deepcopy(manifest), "1" * 64),
    )

    accepted = launcher.validate_agent_artifact(launcher.DEFAULT_ARTIFACT)

    assert accepted.manifest["model"]["id"] == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert (
        accepted.manifest["runtime"]["revision"]
        == "f8dbcd61893702976f9ab03be89c2b9f436d532c"
    )


def test_local_install_requires_digest_before_executing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    assert launcher.EXPECTED_OPENCODE_SHA256 == (
        "ef06e41a35795066e95acde276a42fbbf85d7a683c2787f6a19ed20bcde9b6ff"
    )
    executable = tmp_path / "opencode.exe"
    executable.write_bytes(b"tampered executable")
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "OPENCODE_EXE", executable)
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("a digest-mismatched executable must not run")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden_run)
    with pytest.raises(launcher.LocalCoderError, match="digest mismatch"):
        launcher.validate_local_install(executable)
    assert called is False


def test_local_install_accepts_only_matching_digest_and_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    executable = tmp_path / "opencode.exe"
    content = b"pinned executable fixture"
    executable.write_bytes(content)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "OPENCODE_EXE", executable)
    monkeypatch.setattr(
        launcher, "EXPECTED_OPENCODE_SHA256", hashlib.sha256(content).hexdigest()
    )

    observed: dict[str, object] = {}

    def pinned_version(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="1.18.25\n", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", pinned_version)
    assert launcher.validate_local_install(executable) == executable.resolve()
    assert observed["command"] == [str(executable.resolve()), "--version"]
    assert observed["kwargs"]["cwd"] == str(executable.resolve().parent)
    assert observed["kwargs"]["env"]["NoDefaultCurrentDirectoryInExePath"] == "1"

    executable.write_bytes(content + b"mutation")
    with pytest.raises(launcher.LocalCoderError, match="digest mismatch"):
        launcher.validate_local_install(executable)


def test_local_install_rejects_matching_binary_from_noncanonical_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    project = tmp_path / "project"
    canonical = project / ".local-coder/node_modules/opencode-ai/bin/opencode.exe"
    alternate = tmp_path / "alternate/opencode.exe"
    canonical.parent.mkdir(parents=True)
    alternate.parent.mkdir(parents=True)
    content = b"same pinned bytes"
    canonical.write_bytes(content)
    alternate.write_bytes(content)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)
    monkeypatch.setattr(launcher, "OPENCODE_EXE", canonical)
    monkeypatch.setattr(
        launcher, "EXPECTED_OPENCODE_SHA256", hashlib.sha256(content).hexdigest()
    )

    with pytest.raises(launcher.LocalCoderError, match="canonical project-local"):
        launcher.validate_local_install(alternate)


def test_local_install_rejects_a_reparse_aliased_canonical_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    project = tmp_path / "project"
    canonical = project / ".local-coder/node_modules/opencode-ai/bin/opencode.exe"
    outside = tmp_path / "outside/opencode.exe"
    canonical.parent.mkdir(parents=True)
    outside.parent.mkdir()
    content = b"same pinned bytes"
    outside.write_bytes(content)
    try:
        canonical.symlink_to(outside)
    except OSError:
        pytest.skip("file symlinks are unavailable on this Windows host")
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)
    monkeypatch.setattr(launcher, "OPENCODE_EXE", canonical)
    monkeypatch.setattr(
        launcher, "EXPECTED_OPENCODE_SHA256", hashlib.sha256(content).hexdigest()
    )
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("a reparse-aliased executable must not run")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden_run)
    with pytest.raises(launcher.LocalCoderError, match="symlink|reparse|aliased"):
        launcher.validate_local_install(canonical)
    assert called is False


def test_final_execution_rehash_rejects_every_mutable_client_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    project = tmp_path / "project"
    executable = project / ".local-coder" / "opencode.exe"
    source_config = project / "configs" / "opencode.json"
    isolated_config = project / ".local-coder" / "state" / "config-dir" / "opencode.json"
    source_contract = project / "substrate" / "AGENT_CONTRACT.md"
    installed_contract = (
        project / ".local-coder" / "state" / "xdg-config" / "opencode" / "AGENTS.md"
    )
    for path in (
        executable,
        source_config,
        isolated_config,
        source_contract,
        installed_contract,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    executable_bytes = b"executable"
    config_bytes = b'{"pinned":true}\n'
    contract_bytes = b"contract\n"
    executable.write_bytes(executable_bytes)
    source_config.write_bytes(config_bytes)
    isolated_config.write_bytes(config_bytes)
    source_contract.write_bytes(contract_bytes)
    installed_contract.write_bytes(contract_bytes)
    monkeypatch.setattr(launcher, "PROJECT_ROOT", project)
    monkeypatch.setattr(launcher, "OPENCODE_EXE", executable)
    monkeypatch.setattr(launcher, "SUBSTRATE_CONTRACT", source_contract)
    monkeypatch.setattr(
        launcher, "EXPECTED_OPENCODE_SHA256", hashlib.sha256(executable_bytes).hexdigest()
    )
    monkeypatch.setattr(
        launcher, "EXPECTED_CONTRACT_SHA256", hashlib.sha256(contract_bytes).hexdigest()
    )
    validated = launcher.ValidatedConfig(
        source_config, config_bytes, hashlib.sha256(config_bytes).hexdigest()
    )
    environment = {
        "OPENCODE_CONFIG": str(isolated_config),
        "XDG_CONFIG_HOME": str(installed_contract.parents[1]),
    }
    launcher.verify_execution_inputs(executable, validated, environment)

    for path in (
        executable,
        source_config,
        isolated_config,
        source_contract,
        installed_contract,
    ):
        original = path.read_bytes()
        path.write_bytes(original + b"mutation")
        with pytest.raises(launcher.LocalCoderError, match="changed"):
            launcher.verify_execution_inputs(executable, validated, environment)
        path.write_bytes(original)


@pytest.mark.skipif(
    not (ROOT / ".local-coder/node_modules/opencode-ai/bin/opencode.exe").is_file(),
    reason="project-local OpenCode has not been installed",
)
def test_pinned_client_resolves_only_the_isolated_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launcher = _load_launcher()
    config = launcher.validate_config(CONFIG)
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
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "LOCAL_STATE_ROOT", tmp_path / "isolated-state")
    environment = launcher.isolated_environment(
        config, "http://127.0.0.1:19090"
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
    monkeypatch.setattr(
        launcher, "validate_local_install", lambda _path: executable.resolve()
    )
    monkeypatch.setattr(launcher, "resolve_canonical_artifact", lambda _path: artifact)
    approval = object()
    validated = type("Validated", (), {"server_approval": approval})()
    monkeypatch.setattr(launcher, "validate_agent_artifact", lambda _path: validated)
    monkeypatch.setattr(launcher, "isolated_environment", lambda *_args: {"isolated": "yes"})
    monkeypatch.setattr(launcher, "verify_execution_inputs", lambda *_args: None)
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
    assert server_kwargs["verify_payload_hash"] is True
    assert server_kwargs["artifact_approval"] is approval
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
    monkeypatch.setattr(
        launcher, "validate_local_install", lambda _path: executable.resolve()
    )
    monkeypatch.setattr(launcher, "resolve_canonical_artifact", lambda _path: artifact)
    validated = type("Validated", (), {"server_approval": object()})()
    monkeypatch.setattr(launcher, "validate_agent_artifact", lambda _path: validated)
    monkeypatch.setattr(launcher, "isolated_environment", lambda *_args: {"isolated": "yes"})
    monkeypatch.setattr(launcher, "verify_execution_inputs", lambda *_args: None)
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
    assert 'PinnedArchiveBytes = 62030007' in script
    assert (
        'PinnedArchiveSha256 = "35fe618642f733aa1db8e26a78a1c9ee7cfce47e94cdbd36a37312c9d55e2a45"'
        in script
    )
    assert 'PinnedExecutableBytes = 179651624' in script
    assert (
        'PinnedSha256 = "ef06e41a35795066e95acde276a42fbbf85d7a683c2787f6a19ed20bcde9b6ff"'
        in script
    )
    assert (
        'vendor\\opencode\\opencode-windows-x64-1.18.25.zip"' in script
    )
    assert "Get-VerifiedPlatformOpenCode" in script
    assert "opencode-windows-x64\\bin\\opencode.exe" in script
    assert "opencode-windows-x64-baseline\\bin\\opencode.exe" in script
    assert "function Install-OpenCodeFromBundledArchive" in script
    assert "function Install-OpenCodeFromNpm" in script
    assert "function Install-PinnedOpenCode" in script
    assert "function Invoke-LocalCoderPreflight" in script
    assert "[switch]$FunctionsOnly" in script
    assert "Assert-SafeProjectPath" in script
    assert "FileAttributes]::ReparsePoint" in script
    assert "--prefix $InstallRoot" in script
    assert "--cache $NpmCache" in script
    assert "--save-exact" in script
    assert "--ignore-scripts" in script
    assert " -g " not in script

    pinned_probe = script.split("function Test-PinnedOpenCode", maxsplit=1)[1].split(
        "function Get-VerifiedPlatformOpenCode", maxsplit=1
    )[0]
    first_hash = pinned_probe.index("Get-Sha256File -Path $ExecutablePath")
    execution = pinned_probe.index("& $ExecutablePath --version")
    second_hash = pinned_probe.rindex("Get-Sha256File -Path $ExecutablePath")
    assert first_hash < execution < second_hash

    platform_selection = script.split(
        "function Get-VerifiedPlatformOpenCode", maxsplit=1
    )[1].split("Assert-SafeProjectPath -Path $InstallRoot", maxsplit=1)[0]
    assert platform_selection.index("Get-Sha256File -Path $Candidate") < (
        script.index("[System.IO.File]::Copy")
    )

    archive_install = script.split(
        "function Install-OpenCodeFromBundledArchive", maxsplit=1
    )[1].split("function Install-OpenCodeFromNpm", maxsplit=1)[0]
    assert archive_install.index("$ArchiveItem.Length -ne $PinnedArchiveBytes") < (
        archive_install.index("[System.IO.Compression.ZipFile]::OpenRead")
    )
    assert archive_install.index("$ArchiveSha256 -ne $PinnedArchiveSha256") < (
        archive_install.index("[System.IO.Compression.ZipFile]::OpenRead")
    )
    assert '$Entry.FullName -cne "opencode.exe"' in archive_install
    assert "$Archive.Entries.Count -ne 1" in archive_install

    installer = script.split("function Install-PinnedOpenCode", maxsplit=1)[1]
    archive_branch = installer.index(
        "if (Test-Path -LiteralPath $BundledArchive -PathType Leaf)"
    )
    bundled_call = installer.index("Install-OpenCodeFromBundledArchive")
    npm_call = installer.index("Install-OpenCodeFromNpm")
    assert archive_branch < bundled_call < npm_call


def test_bundled_archive_installs_only_the_verified_single_entry(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    archive = project / "vendor/opencode/opencode-windows-x64-1.18.25.zip"
    archive.parent.mkdir(parents=True)
    executable_bytes = b"small deterministic OpenCode fixture"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("opencode.exe", executable_bytes)

    completed = _run_bundled_archive_installer(project, archive, executable_bytes)

    assert completed.returncode == 0, completed.stderr
    installed = project / ".local-coder/node_modules/opencode-ai/bin/opencode.exe"
    assert installed.read_bytes() == executable_bytes


@pytest.mark.parametrize("mutation", ["digest", "extra-entry"])
def test_bundled_archive_rejects_tamper_without_npm_fallback(
    tmp_path: Path, mutation: str
) -> None:
    project = tmp_path / "project"
    archive = project / "vendor/opencode/opencode-windows-x64-1.18.25.zip"
    archive.parent.mkdir(parents=True)
    executable_bytes = b"small deterministic OpenCode fixture"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("opencode.exe", executable_bytes)
        if mutation == "extra-entry":
            bundle.writestr("unexpected.txt", b"must be rejected")
    expected_digest = "0" * 64 if mutation == "digest" else None

    completed = _run_bundled_archive_installer(
        project,
        archive,
        executable_bytes,
        archive_sha256=expected_digest,
    )

    assert completed.returncode != 0
    assert not (
        project / ".local-coder/node_modules/opencode-ai/bin/opencode.exe"
    ).exists()
