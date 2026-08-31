from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_substrate_coder.py"


def _load_module():
    name = "_test_benchmark_substrate_coder"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()


def _tool(name: str, arguments: dict, *, status: str = "completed") -> dict:
    return {
        "type": "tool_use",
        "part": {
            "tool": name,
            "state": {"status": status, "input": arguments},
        },
    }


def _valid_tool_events() -> list[dict]:
    events = [
        _tool("read", {"path": "README.md"}),
        _tool("grep", {"pattern": "FiniteConeSDF", "path": "src"}),
        _tool("edit", {"path": "src/substrate_app.py", "oldString": "TODO", "newString": "done"}),
    ]
    events.extend(
        _tool("bash", {"command": command, "workdir": "."})
        for command in (
            gate.APPLICATION_COMMAND,
            gate.RENDER_COMMAND,
            gate.PYTEST_COMMAND,
            gate.VALIDATE_COMMAND,
        )
    )
    return events


def _valid_replay(generations: int = 5) -> dict:
    definition_hashes = [f"sha256:{index:064x}" for index in range(1, 5)]
    rows = []
    for generation in range(generations):
        status = "INDETERMINATE" if generation == 1 else "VERIFIED"
        crossing = "INDETERMINATE" if generation == 1 else "TRUE"
        rows.append(
            {
                "generation": generation,
                "feedback": {
                    "source_generation": generation,
                    "target_generation": generation + 1,
                },
                "address": {
                    "kind": "log-polar-lut",
                    "rho": generation / 10,
                    "theta": generation / 20,
                },
                "packing": {
                    "widths": dict(gate.KEY_WIDTHS),
                    "contiguous": generation,
                    "morton": generation + 10,
                    "contiguous_round_trip": True,
                    "morton_round_trip": True,
                },
                "bits": {
                    "payload_parity_bit": 0,
                    "topology_orientation_bit": 1,
                    "jitter_control_bit": generation & 1,
                    "branch_predicate_bit": (generation + 1) & 1,
                },
                "bit_role_provenance": {
                    "payload_parity_bit": "xor_parity(payload)",
                    "topology_orientation_bit": "orientation_reversals mod 2",
                    "jitter_control_bit": "OneBitJitter.bit",
                    "branch_predicate_bit": "explicit routing predicate",
                },
                "vectors": [
                    {"role": "velocity", "origin": [0, 0, 0], "displacement": [1, 0, 0]},
                    {"role": "acceleration", "origin": [0, 0, 0], "displacement": [0, 1, 0]},
                ],
                "kinematics": {
                    "position": [generation, 0.0, 0.0],
                    "velocity": [1.0, 0.0, 0.0],
                    "acceleration": [0.0, 0.0, 0.0],
                },
                "geometry": {
                    "finite_cone": {"kind": "exact-finite-cone-sdf", "distance": -0.25},
                    "sphere": {"kind": "exact-sphere-sdf", "distance": 0.5},
                },
                "event": {
                    "status": status,
                    "support": "TRUE",
                    "compatibility": "TRUE",
                    "crossing": crossing,
                },
                "state_digest": f"sha256:{100 + generation:064x}",
                "lineage_digest": f"{200 + generation:064x}",
            }
        )
    return {
        "format": gate.REPLAY_FORMAT,
        "application_id": gate.APPLICATION_ID,
        "requested_generations": generations,
        "key_layout": {"widths": dict(gate.KEY_WIDTHS), "total_bits": 64},
        "definition_graph": {
            "content_hash": f"sha256:{999:064x}",
            "definition_hashes": definition_hashes,
            "feedback": {
                "source_generation": 0,
                "target_generation": 1,
                "fixed_point_claim": False,
            },
        },
        "generations": rows,
        "lineage_head": rows[-1]["lineage_digest"],
        "distinctions": {"self_reference_may_promote": False},
    }


def test_gate_reuses_exact_pinned_launcher_identity() -> None:
    assert gate.MODEL_ID == "local-runtime/local-qwen3-30b-a3b"
    assert gate.PINNED_OPENCODE_VERSION == "1.18.25"
    assert gate._BASE._LAUNCHER.MODEL_ID == gate.MODEL_ID
    assert gate._BASE._LAUNCHER.PINNED_OPENCODE_VERSION == gate.PINNED_OPENCODE_VERSION


def test_scope_and_prompt_make_the_bounded_claim_honestly() -> None:
    prompt = gate._agent_prompt()
    lowered = prompt.lower()
    assert "not a broad-intelligence or compression proof" in lowered
    assert "retrospective" in lowered
    assert "not a preventive sandbox" in lowered
    assert "legacy/archive" in lowered
    assert "n to n+1" in lowered
    assert "may only propose quarantined extensions and cannot promote" in lowered
    for command in gate.ALLOWED_BASH_COMMANDS:
        assert f"`{command}`" in prompt


@pytest.mark.parametrize(
    "path",
    (
        "../legacy/code.py",
        "legacy/code.py",
        "archive/code.py",
        "C:/outside.py",
        "/outside.py",
        " spaced.py",
    ),
)
def test_clean_room_relative_paths_fail_closed(path: str) -> None:
    with pytest.raises(gate.GateError):
        gate._safe_relative(path, label="test")


def test_copy_allowlist_is_explicit_and_contains_no_legacy_or_archive() -> None:
    plan = gate._copy_plan()
    destinations = [destination for _, destination in plan]
    assert len(plan) == 13
    assert len(destinations) == len(set(destinations))
    assert "src/nhdf_edge/substrate_runtime.py" in destinations
    assert "src/nhdf_edge/substrate_graph.py" in destinations
    assert "src/nhdf_edge/substrate_contract.py" in destinations
    assert "src/nhdf_edge/substrate_pdf.py" in destinations
    assert gate.PARENT_PROVENANCE_PDF in destinations
    assert all("legacy" not in value.lower() and "archive" not in value.lower() for value in destinations)


def test_fixture_copies_bound_sources_and_starts_as_failing_clean_commit(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    metadata = gate._create_fixture(fixture, require_tracked=False)
    assert metadata["legacy_or_archive_material_loaded"] is False
    assert len(metadata["copied_sources"]) == 13
    assert gate._BASE._run_git(fixture, ["status", "--porcelain"]) == b""
    assert gate._BASE._run_git(fixture, ["rev-parse", "HEAD"]).decode().strip() == metadata["initial_head"]
    for record in metadata["copied_sources"]:
        copied = fixture / record["destination"]
        assert copied.stat().st_size == record["bytes"]
        assert gate._sha256_file(copied) == record["sha256"]
    result = gate._BASE._run_process(
        [sys.executable, "-m", "pytest", "-q"], cwd=fixture, timeout=30.0
    )
    assert result.returncode != 0


def test_manifest_template_is_bound_but_deliberately_unsealed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    gate._create_fixture(fixture, require_tracked=False)
    template = json.loads((fixture / "app/application-manifest.template.json").read_text(encoding="utf-8"))
    registry = json.loads((fixture / "substrate/profiles/registry.json").read_text(encoding="utf-8"))
    profile = next(item for item in registry["profiles"] if item["profile_id"] == gate.PROFILE_ID)
    assert template["kernel"]["sha256"] == gate._sha256_file(fixture / template["kernel"]["path"])
    assert template["profiles"] == [{"profile_id": gate.PROFILE_ID, "sha256": profile["sha256"]}]
    assert set(template["mappings"]) == set(gate.MAPPING_CATEGORIES)
    assert template["self_reference"] == {
        "enabled": True,
        "bounded_generations": 5,
        "may_propose_extensions": True,
        "may_promote_extensions": False,
        "proposal_disposition": "QUARANTINED",
    }
    assert template["evidence"][0]["sha256"] == "0" * 64


@pytest.mark.parametrize(
    "command",
    (
        "python -m pytest",
        "python -m pytest -q && curl https://example.com",
        "python -m pytest -q; git commit -am x",
        "pip install reportlab",
        "powershell -Command Invoke-WebRequest http://localhost",
        "git status",
    ),
)
def test_bash_allowlist_rejects_variants_and_network_or_git_commands(command: str) -> None:
    with pytest.raises(gate.GateError):
        gate._validate_bash_command(command)


def test_exact_four_bash_commands_are_accepted() -> None:
    assert {gate._validate_bash_command(command) for command in gate.ALLOWED_BASH_COMMANDS} == set(
        gate.ALLOWED_BASH_COMMANDS
    )


def test_recorded_tool_audit_accepts_only_expected_offline_workflow(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    result = gate._evaluate_tool_events(_valid_tool_events(), fixture)
    assert result["passed"] is True
    assert result["bash_commands_in_order"] == [
        gate.APPLICATION_COMMAND,
        gate.RENDER_COMMAND,
        gate.PYTEST_COMMAND,
        gate.VALIDATE_COMMAND,
    ]
    assert result["audit_timing"] == "POST_RUN_RETROSPECTIVE"
    assert result["preventive_sandbox"] is False


def test_recorded_tool_audit_rejects_outside_read_path(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    events = _valid_tool_events()
    events[0] = _tool("read", {"path": "../secret.txt"})
    with pytest.raises(gate.GateError, match="escaped"):
        gate._evaluate_tool_events(events, fixture)


def test_recorded_tool_audit_rejects_immutable_edit(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    events = _valid_tool_events()
    events[2] = _tool("edit", {"path": "tests/test_substrate_app.py"})
    with pytest.raises(gate.GateError, match="immutable"):
        gate._evaluate_tool_events(events, fixture)


def test_recorded_tool_audit_rejects_hostile_glob(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    events = _valid_tool_events()
    events[1] = _tool("glob", {"pattern": "../**/*"})
    with pytest.raises(gate.GateError, match="inside"):
        gate._evaluate_tool_events(events, fixture)


def test_recorded_tool_audit_rejects_reordered_final_commands(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    events = _valid_tool_events()
    events[-1], events[-2] = events[-2], events[-1]
    with pytest.raises(gate.GateError, match="in order"):
        gate._evaluate_tool_events(events, fixture)


def test_recorded_tool_audit_allows_exact_development_reruns_before_final_sequence(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    events = _valid_tool_events()
    events.insert(3, _tool("bash", {"command": gate.PYTEST_COMMAND, "workdir": "."}))
    result = gate._evaluate_tool_events(events, fixture)
    assert result["bash_commands_in_order"][-4:] == [
        gate.APPLICATION_COMMAND,
        gate.RENDER_COMMAND,
        gate.PYTEST_COMMAND,
        gate.VALIDATE_COMMAND,
    ]


def test_recorded_tool_audit_rejects_failed_or_external_tool(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    with pytest.raises(gate.GateError, match="failed"):
        gate._evaluate_tool_events([*_valid_tool_events(), _tool("read", {"path": "README.md"}, status="error")], fixture)
    with pytest.raises(gate.GateError, match="disallowed"):
        gate._evaluate_tool_events([*_valid_tool_events(), _tool("webfetch", {"url": "https://example.com"})], fixture)


def test_replay_evaluator_accepts_exact_bounded_primitive_evidence() -> None:
    result = gate._validate_replay_payload(_valid_replay(), expected_generations=5)
    assert result["passed"] is True
    assert result["generations"] == 5
    assert result["exact_key_widths"] == gate.KEY_WIDTHS
    assert result["statuses"] == ["INDETERMINATE", "VERIFIED"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["key_layout"].update(widths={"rho": 21, "theta": 17, "time": 14, "phi": 12}), "20/18/14/12"),
        (lambda value: value["definition_graph"]["feedback"].update(target_generation=2), "generation 0 to 1"),
        (lambda value: value["generations"][0]["packing"].update(contiguous_round_trip=False), "round trips"),
        (lambda value: value["generations"][0]["geometry"].pop("finite_cone"), "finite cone"),
        (lambda value: value["generations"][0]["bit_role_provenance"].update(jitter_control_bit="xor_parity(payload)"), "bit-role provenance"),
        (lambda value: value["generations"][0]["feedback"].update(target_generation=4), r"n to n\+1"),
        (lambda value: value["generations"][0].update(state_digest="bad"), "state digest"),
        (lambda value: value["distinctions"].update(self_reference_may_promote=True), "promotion"),
    ),
)
def test_replay_evaluator_fails_closed_on_semantic_corruption(mutation, message: str) -> None:
    replay = _valid_replay()
    mutation(replay)
    with pytest.raises(gate.GateError, match=message):
        gate._validate_replay_payload(replay, expected_generations=5)


def test_replay_evaluator_rejects_nonfinite_numbers_and_missing_tristate() -> None:
    replay = _valid_replay()
    replay["generations"][0]["address"]["rho"] = float("nan")
    with pytest.raises(gate.GateError, match="finite"):
        gate._validate_replay_payload(replay, expected_generations=5)
    replay = _valid_replay()
    for row in replay["generations"]:
        row["event"] = {
            "status": "VERIFIED",
            "support": "TRUE",
            "compatibility": "TRUE",
            "crossing": "TRUE",
        }
    with pytest.raises(gate.GateError, match="indeterminate"):
        gate._validate_replay_payload(replay, expected_generations=5)


def test_pdf_sidecar_evaluator_requires_digest_source_and_validation(tmp_path: Path) -> None:
    fixture = tmp_path
    report = fixture / "report/substrate-report-v0.1.md"
    pdf = fixture / "output/substrate-report.pdf"
    sidecar = fixture / "output/substrate-report.pdf.metadata.json"
    report.parent.mkdir(parents=True)
    pdf.parent.mkdir(parents=True)
    report.write_text("version: 0.1.0\n", encoding="utf-8")
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    sidecar.write_text(
        json.dumps(
            {
                "source": {"sha256": gate._sha256_file(report)},
                "output": {"sha256": gate._sha256_file(pdf), "pages": 1},
                "validation": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    assert gate._evaluate_pdf(fixture)["reopened"] is True
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["output"]["sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(gate.GateError, match="digest"):
        gate._evaluate_pdf(fixture)


def test_git_evaluator_accepts_only_exact_unstaged_artifact_set(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    baseline = gate._create_fixture(fixture, require_tracked=False)
    for relative in gate.MUTABLE_TRACKED_PATHS:
        path = fixture / relative
        path.write_bytes(path.read_bytes() + b"\nchanged\n")
    for relative in gate.GENERATED_PATHS:
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")
    result = gate._evaluate_git_state(fixture, baseline)
    assert result["exact_head_unchanged"] is True
    assert result["only_expected_paths_changed_or_created"] is True
    (fixture / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(gate.GateError, match="exact allowlist"):
        gate._evaluate_git_state(fixture, baseline)


def test_git_evaluator_rejects_staged_expected_change(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    baseline = gate._create_fixture(fixture, require_tracked=False)
    for relative in gate.MUTABLE_TRACKED_PATHS:
        path = fixture / relative
        path.write_bytes(path.read_bytes() + b"\nchanged\n")
    for relative in gate.GENERATED_PATHS:
        path = fixture / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")
    gate._BASE._run_git(fixture, ["add", "--", "src/substrate_app.py"])
    with pytest.raises(gate.GateError, match="staged"):
        gate._evaluate_git_state(fixture, baseline)


def test_environment_reuses_isolation_and_scrubs_credential_like_parent_values(tmp_path: Path) -> None:
    config_path, config_bytes, _ = gate._BASE._load_pinned_config(
        PROJECT_ROOT / "configs/opencode_nhdf_local.json"
    )
    copied = tmp_path / "isolated/config/opencode/opencode.json"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(config_bytes)
    environment = gate._BASE._agent_environment(
        {
            "PATH": "C:\\Windows\\System32",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "GITHUB_TOKEN": "token",
            "OPENAI_API_KEY": "key",
            "DATABASE_URL": "postgres://credential",
        },
        tmp_path / "isolated",
        copied,
        server_url="http://127.0.0.1:1234",
    )
    assert environment["PATH"] == "C:\\Windows\\System32"
    for secret in ("AWS_SECRET_ACCESS_KEY", "GITHUB_TOKEN", "OPENAI_API_KEY", "DATABASE_URL"):
        assert secret not in environment
    assert environment["UGTOMS_LOCAL_CODER_BASE_URL"] == "http://127.0.0.1:1234"
    assert environment["UGTOMS_SUBSTRATE_CONTRACT_SHA256"] == gate._BASE._LAUNCHER.EXPECTED_CONTRACT_SHA256
    assert Path(environment["XDG_CONFIG_HOME"], "opencode", "AGENTS.md").is_file()


@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:1234",
        "http://example.com:1234",
        "http://user:pass@localhost:1234",
        "http://127.0.0.1:1234/other",
        "http://127.0.0.1:1234/?token=x",
    ),
)
def test_existing_server_must_be_exact_credential_free_loopback(url: str) -> None:
    with pytest.raises(gate.GateError):
        gate._BASE._normalize_server_url(url)


def test_config_rejects_any_extra_or_hostile_endpoint() -> None:
    _, _, config = gate._BASE._load_pinned_config(PROJECT_ROOT / "configs/opencode_nhdf_local.json")
    hostile = copy.deepcopy(config)
    hostile["provider"]["evil"] = {
        "npm": "@ai-sdk/openai-compatible",
        "options": {"baseURL": "https://example.com/v1"},
        "models": {},
    }
    with pytest.raises(gate.GateError):
        gate._BASE._validate_opencode_config(hostile, "http://localhost:1234")


def test_opencode_version_validator_requires_exact_pin() -> None:
    good = subprocess.CompletedProcess(["opencode"], 0, b"1.18.25\n", b"")
    assert gate._BASE._validated_opencode_version(good) == "1.18.25"
    wrong = subprocess.CompletedProcess(["opencode"], 0, b"1.18.26\n", b"")
    with pytest.raises(gate.GateError, match="1.18.25"):
        gate._BASE._validated_opencode_version(wrong)


def test_cli_refuses_nonpositive_timeouts() -> None:
    with pytest.raises(SystemExit):
        gate.main(
            [
                "--server-url",
                "http://127.0.0.1:1234",
                "--opencode-exe",
                "missing",
                "--config",
                "missing",
                "--output-root",
                "missing",
                "--agent-timeout-seconds",
                "0",
            ]
        )
