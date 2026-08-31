from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_substrate_repair.py"


def _load_gate() -> ModuleType:
    name = "benchmark_substrate_repair_under_test"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate() -> ModuleType:
    return _load_gate()


def _tool(name: str, arguments: dict, *, status: str = "completed") -> dict:
    return {
        "type": "tool",
        "name": name,
        "status": status,
        "input": arguments,
    }


def _valid_tool_events(gate: ModuleType) -> list[dict]:
    events = [
        _tool("read", {"filePath": path})
        for path in sorted(gate.REQUIRED_READ_PATHS)
    ]
    events.extend(
        [
            _tool(
                "grep",
                {
                    "pattern": "target_generation|FeedbackEdge",
                    "path": ".",
                    "include": "*.py",
                },
            ),
            _tool(
                "edit",
                {
                    "filePath": gate.EXPECTED_SOURCE,
                    "oldString": gate._BROKEN_FEEDBACK_LINE,
                    "newString": gate._FIXED_FEEDBACK_LINE,
                },
            ),
            _tool("bash", {"command": gate.PYTEST_COMMAND, "workdir": "."}),
        ]
    )
    return events


def _fixture(gate: ModuleType, tmp_path: Path) -> tuple[Path, dict]:
    fixture = tmp_path / "fixture"
    baseline = gate._create_fixture(fixture, require_tracked=False)
    return fixture, baseline


def _repair(gate: ModuleType, fixture: Path) -> None:
    (fixture / gate.EXPECTED_SOURCE).write_text(
        gate._fixed_application(), encoding="utf-8", newline="\n"
    )


def _run_fixture_pytest(gate: ModuleType, fixture: Path) -> subprocess.CompletedProcess:
    return gate._BASE._run_process(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=fixture,
        env=gate._fixture_environment(),
        timeout=30.0,
    )


def test_gate_reuses_canonical_local_identity_and_narrow_scope(gate: ModuleType) -> None:
    assert gate.MODEL_ID == gate._BASE._LAUNCHER.MODEL_ID
    assert gate.PINNED_OPENCODE_VERSION == "1.18.25"
    assert gate._BASE._LAUNCHER.EXPECTED_OPENCODE_SHA256 == (
        "ef06e41a35795066e95acde276a42fbbf85d7a683c2787f6a19ed20bcde9b6ff"
    )
    assert gate.MIN_CONTEXT_TOKENS == 32_768
    text = (gate.__doc__ or "") + gate._agent_prompt() + gate._parser().description
    for phrase in (
        "focused",
        "not a broad",
        "compression",
        "general coding competence",
        "retrospective",
        "not a preventive sandbox",
    ):
        assert phrase.lower() in text.lower()


def test_fixture_source_contains_exactly_one_semantic_defect(gate: ModuleType) -> None:
    broken = gate._broken_application()
    fixed = gate._fixed_application()
    assert broken.count(gate._BROKEN_FEEDBACK_LINE) == 1
    assert gate._BROKEN_FEEDBACK_LINE not in fixed
    assert fixed.count(gate._FIXED_FEEDBACK_LINE) == 1
    broken_lines = broken.splitlines()
    fixed_lines = fixed.splitlines()
    differences = [
        (before, after)
        for before, after in zip(broken_lines, fixed_lines)
        if before != after
    ]
    assert differences == [
        (gate._BROKEN_FEEDBACK_LINE, gate._FIXED_FEEDBACK_LINE)
    ]


@pytest.mark.parametrize(
    "path",
    (
        "../outside.py",
        "legacy/module.py",
        "archive/data.json",
        "archives/data.json",
        "C:/outside.py",
        "/outside.py",
        "file:///outside.py",
    ),
)
def test_clean_room_paths_reject_escape_and_legacy_archive(
    gate: ModuleType, path: str
) -> None:
    with pytest.raises(gate.GateError, match="allowlist"):
        gate._safe_relative(path, label="test path")


def test_copy_allowlist_is_small_real_and_profile_bound(
    gate: ModuleType, tmp_path: Path
) -> None:
    assert set(gate.COPIED_PROJECT_FILES) == {
        "src/nhdf_edge/substrate_graph.py",
        "src/nhdf_edge/substrate_runtime.py",
        "substrate/kernel/contract.json",
        "substrate/profiles/registry.json",
        "substrate/profiles/sclp-foundational.json",
    }
    fixture, baseline = _fixture(gate, tmp_path)
    binding = baseline["contract_profile_binding"]
    assert binding["kernel_id"] == "ugtoms-kernel-v0.1"
    assert binding["profile_id"] == "sclp-foundational"
    assert binding["automatic_promotion"] is False
    assert len(baseline["copied_sources"]) == len(gate.COPIED_PROJECT_FILES)
    assert all(record["sha256"] == gate._sha256_file(fixture / record["destination"])
               for record in baseline["copied_sources"])
    assert baseline["legacy_or_archive_material_loaded"] is False


def test_fixture_starts_failing_only_at_declared_feedback_defect(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, baseline = _fixture(gate, tmp_path)
    result = _run_fixture_pytest(gate, fixture)
    output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
    assert result.returncode != 0
    assert "feedback must cross exactly one generation" in output
    assert baseline["defect"] == {
        "kind": "same_generation_feedback",
        "broken": "n -> n",
        "required": "n -> n + 1",
        "defect_count": 1,
    }
    assert gate._BASE._run_git(
        fixture, ["status", "--porcelain=v1", "--untracked-files=all"]
    ) == b""


def test_exact_repair_passes_tests_two_replays_and_git_audit(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, baseline = _fixture(gate, tmp_path)
    _repair(gate, fixture)
    result = _run_fixture_pytest(gate, fixture)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(
        "utf-8", errors="replace"
    )
    replay = gate._evaluate_replay(
        fixture,
        tmp_path / "replays",
        timeout=30.0,
        environment=gate._fixture_environment(),
    )
    assert replay["byte_deterministic_twice"] is True
    assert replay["independent_runs"] == 2
    assert replay["explicit_next_generation_feedback"] is True
    assert replay["exact_key_widths"] == gate.KEY_WIDTHS
    git = gate._evaluate_git_state(fixture, baseline)
    assert git["exact_head_unchanged"] is True
    assert git["changed_paths"] == [gate.EXPECTED_SOURCE]
    assert git["exact_one_line_semantic_repair"] is True


def test_recorded_tool_audit_accepts_only_the_focused_offline_workflow(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, _ = _fixture(gate, tmp_path)
    result = gate._evaluate_tool_events(_valid_tool_events(gate), fixture)
    assert result["passed"] is True
    assert result["required_read_paths"] == sorted(gate.REQUIRED_READ_PATHS)
    assert result["exact_edit_calls"] == 1
    assert result["exact_bash_calls"] == 1
    assert result["pytest_was_final_recorded_tool"] is True
    assert result["recorded_network_package_git_commands"] is False
    assert result["recorded_legacy_or_archive_access"] is False
    assert result["audit_timing"] == "POST_RUN_RETROSPECTIVE"
    assert result["preventive_sandbox"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda events, gate: events[0]["input"].update(
                filePath="../outside.json"
            ),
            "escaped",
        ),
        (
            lambda events, gate: events[0]["input"].update(
                filePath="legacy/secret.json"
            ),
            "legacy/archive",
        ),
        (
            lambda events, gate: next(
                event for event in events if event["name"] == "grep"
            )["input"].update(path=".."),
            "escaped",
        ),
        (
            lambda events, gate: next(
                event for event in events if event["name"] == "grep"
            )["input"].update(include="../*.py"),
            "inside the disposable fixture",
        ),
        (
            lambda events, gate: events.insert(
                -1,
                _tool("glob", {"pattern": "legacy/**/*.py", "path": "."})
            ),
            "legacy/archive",
        ),
        (
            lambda events, gate: next(
                event for event in events if event["name"] == "edit"
            )["input"].update(filePath="tests/test_sclp_repair.py"),
            "other than",
        ),
        (
            lambda events, gate: next(
                event for event in events if event["name"] == "bash"
            )["input"].update(command="pip install pytest"),
            "exactly",
        ),
        (
            lambda events, gate: events.append(
                _tool("bash", {"command": gate.PYTEST_COMMAND})
            ),
            "exactly one recorded Bash",
        ),
        (
            lambda events, gate: events.append(
                _tool(
                    "edit",
                    {
                        "filePath": gate.EXPECTED_SOURCE,
                        "oldString": "x",
                        "newString": "y",
                    },
                )
            ),
            "exactly one recorded Edit",
        ),
        (
            lambda events, gate: events.append(
                _tool("web", {"url": "https://example.invalid"})
            ),
            "disallowed",
        ),
        (
            lambda events, gate: events.append(
                _tool("todowrite", {"todos": []})
            ),
            "final recorded tool",
        ),
        (
            lambda events, gate: next(
                event for event in events if event["name"] == "read"
            ).update(status="failed"),
            "failed or incomplete",
        ),
    ),
)
def test_tool_audit_rejects_hostile_or_expansive_events(
    gate: ModuleType, tmp_path: Path, mutation, message: str
) -> None:
    fixture, _ = _fixture(gate, tmp_path)
    events = _valid_tool_events(gate)
    mutation(events, gate)
    with pytest.raises(gate.GateError, match=message):
        gate._evaluate_tool_events(events, fixture)


def test_tool_audit_requires_explicit_contract_profile_and_source_reads(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, _ = _fixture(gate, tmp_path)
    events = _valid_tool_events(gate)
    events = [
        event
        for event in events
        if event.get("input", {}).get("filePath")
        != "substrate/profiles/sclp-foundational.json"
    ]
    with pytest.raises(gate.GateError, match="did not inspect required"):
        gate._evaluate_tool_events(events, fixture)


@pytest.fixture
def valid_replay(gate: ModuleType, tmp_path: Path) -> dict:
    fixture, _ = _fixture(gate, tmp_path)
    _repair(gate, fixture)
    output = tmp_path / "replay.json"
    payload = gate._run_replay(
        fixture,
        output,
        timeout=30.0,
        environment=gate._fixture_environment(),
    )
    gate._validate_replay_payload(payload)
    return payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda value: value["definition_graph"]["feedback"].update(
                target_generation=0
            ),
            "generation 0 to 1",
        ),
        (
            lambda value: value["generations"][1]["feedback"].update(
                target_generation=1
            ),
            r"not handed to n \+ 1",
        ),
        (
            lambda value: value["generations"][0]["packing"].update(
                widths={"rho": 21, "theta": 17, "time": 14, "phi": 12}
            ),
            "20/18/14/12",
        ),
        (
            lambda value: value["definition_graph"].update(
                definition_hashes=["sha256:" + "1" * 64] * 2
            ),
            "not unique",
        ),
        (
            lambda value: value["generations"][0].update(jitter_bit=True),
            "jitter bit",
        ),
        (
            lambda value: value["generations"][0].update(state_digest="bad"),
            "state hash",
        ),
        (
            lambda value: value["self_reference"].update(may_promote=True),
            "promotion",
        ),
    ),
)
def test_replay_evaluator_fails_closed_on_semantic_tamper(
    gate: ModuleType, valid_replay: dict, mutation, message: str
) -> None:
    payload = copy.deepcopy(valid_replay)
    mutation(payload)
    with pytest.raises(gate.GateError, match=message):
        gate._validate_replay_payload(payload)


def test_git_audit_rejects_unexpected_or_ignored_extra_files(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, baseline = _fixture(gate, tmp_path)
    _repair(gate, fixture)
    hidden = fixture / "__pycache__/sneaky.pyc"
    hidden.parent.mkdir(parents=True)
    hidden.write_bytes(b"unexpected")
    with pytest.raises(gate.GateError, match="immutable or undeclared"):
        gate._evaluate_git_state(fixture, baseline)


def test_git_audit_rejects_staged_source_and_immutable_tamper(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, baseline = _fixture(gate, tmp_path)
    _repair(gate, fixture)
    gate._BASE._run_git(fixture, ["add", "--", gate.EXPECTED_SOURCE])
    with pytest.raises(gate.GateError, match="staged"):
        gate._evaluate_git_state(fixture, baseline)

    fixture_two, baseline_two = _fixture(gate, tmp_path / "second")
    _repair(gate, fixture_two)
    (fixture_two / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(gate.GateError, match="exactly one unstaged"):
        gate._evaluate_git_state(fixture_two, baseline_two)


def test_git_audit_rejects_nonminimal_change_inside_allowed_source(
    gate: ModuleType, tmp_path: Path
) -> None:
    fixture, baseline = _fixture(gate, tmp_path)
    source = gate._fixed_application().replace(
        '"""Mostly-correct bounded SCLP repair fixture with one feedback defect."""',
        '"""Reformatted even though the semantic defect was repaired."""',
    )
    (fixture / gate.EXPECTED_SOURCE).write_text(
        source, encoding="utf-8", newline="\n"
    )
    with pytest.raises(gate.GateError, match="exactly the one-line"):
        gate._evaluate_git_state(fixture, baseline)


def test_canonical_config_rejects_an_exact_copy_at_another_path(
    gate: ModuleType, tmp_path: Path
) -> None:
    canonical = gate._BASE._LAUNCHER.DEFAULT_CONFIG
    resolved, raw, config = gate._load_canonical_config(canonical)
    assert resolved == canonical.resolve()
    assert config["model"] == gate.MODEL_ID
    copied = tmp_path / "copied-config.json"
    copied.write_bytes(raw)
    with pytest.raises(gate.GateError, match="only the canonical"):
        gate._load_canonical_config(copied)


def test_canonical_opencode_rejects_same_bytes_from_another_path(
    gate: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = tmp_path / "project/opencode.exe"
    alternate = tmp_path / "alternate/opencode.exe"
    canonical.parent.mkdir(parents=True)
    alternate.parent.mkdir(parents=True)
    canonical.write_bytes(b"pinned")
    alternate.write_bytes(b"pinned")
    monkeypatch.setattr(gate._BASE._LAUNCHER, "OPENCODE_EXE", canonical)
    monkeypatch.setattr(
        gate._BASE._LAUNCHER, "validate_local_install", lambda _path: None
    )
    assert gate._resolve_canonical_opencode(canonical) == canonical.resolve()
    with pytest.raises(gate.GateError, match="only the canonical"):
        gate._resolve_canonical_opencode(alternate)


def test_model_gate_requires_loopback_served_32k_metadata(gate: ModuleType) -> None:
    result = gate._BASE._validate_models_payload(
        {
            "data": [
                {
                    "id": gate.MODEL_ALIAS,
                    "meta": {"n_ctx": 32_768, "n_ctx_train": 262_144},
                }
            ]
        },
        minimum_context=gate.MIN_CONTEXT_TOKENS,
    )
    assert result["reported_context_tokens"] == 32_768
    with pytest.raises(gate.GateError, match="need at least"):
        gate._BASE._validate_models_payload(
            {"data": [{"id": gate.MODEL_ALIAS, "meta": {"n_ctx": 8192}}]},
            minimum_context=gate.MIN_CONTEXT_TOKENS,
        )


def test_parser_uses_canonical_defaults_and_requires_positive_timeouts(
    gate: ModuleType,
) -> None:
    args = gate._parser().parse_args(
        [
            "--server-url",
            "http://127.0.0.1:18084",
            "--output-root",
            "evidence/substrate-repair-agent",
        ]
    )
    assert Path(args.opencode_exe) == gate._BASE._LAUNCHER.OPENCODE_EXE
    assert Path(args.config) == gate._BASE._LAUNCHER.DEFAULT_CONFIG
    assert args.agent_timeout_seconds == 600.0
    with pytest.raises(SystemExit):
        gate.main(
            [
                "--server-url",
                "http://127.0.0.1:18084",
                "--output-root",
                "unused",
                "--fixture-timeout-seconds",
                "0",
            ]
        )


def test_evidence_format_and_prompt_forbid_expansive_work(gate: ModuleType) -> None:
    prompt = gate._agent_prompt()
    assert gate.FORMAT == "ugtoms-substrate-repair-gate-0.1"
    assert "Edit exactly once" in prompt
    assert gate.EXPECTED_SOURCE in prompt
    assert f"`{gate.PYTEST_COMMAND}`" in prompt
    for forbidden in (
        "Do not create files",
        "stage, commit",
        "network/web/package installation",
        "legacy/archive",
    ):
        assert forbidden in prompt
