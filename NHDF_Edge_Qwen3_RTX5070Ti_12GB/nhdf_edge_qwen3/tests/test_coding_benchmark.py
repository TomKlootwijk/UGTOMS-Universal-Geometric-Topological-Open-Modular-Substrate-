from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from email.message import Message
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_hybrid_coding.py"
SPEC = importlib.util.spec_from_file_location("benchmark_hybrid_coding", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _task(task_id: str):
    return next(task for task in benchmark.TASKS if task["id"] == task_id)


def test_extracts_plain_and_single_fenced_python() -> None:
    source = "def clamp(value, low, high):\n    return max(low, min(value, high))"
    assert benchmark._extract_python(source) == source
    assert benchmark._extract_python(f"```python\n{source}\n```") == source
    with pytest.raises(ValueError, match="exactly one"):
        benchmark._extract_python(f"```python\n{source}\n```\n```python\n{source}\n```")


def test_safety_gate_accepts_normal_bounded_constructs_and_rejects_other_imports() -> None:
    task = _task("clamp_integer")
    benchmark._validate_source(
        "def clamp(value, low, high):\n"
        "    return low if value < low else high if value > high else value\n",
        task,
    )
    with pytest.raises(ValueError, match="forbidden syntax: Import"):
        benchmark._validate_source(
            "def clamp(value, low, high):\n    import os\n    return value\n", task
        )
    benchmark._validate_source(
        "def clamp(value, low, high):\n"
        "    values = sorted([value, low, high], key=lambda item: item)\n"
        "    while len(values) > 3:\n"
        "        values.pop()\n"
        "    return values[1]\n",
        task,
    )
    with pytest.raises(ValueError, match="only `from collections import deque` is allowed"):
        benchmark._validate_source(
            "def clamp(value, low, high):\n"
            "    from os import getenv\n"
            "    return value\n",
            task,
        )


def test_machine_scoring_executes_exact_tests_in_isolated_child() -> None:
    task = _task("stable_deduplication")
    passing = benchmark._score_source(
        "def stable_unique(values):\n"
        "    result = []\n"
        "    for value in values:\n"
        "        if value not in result:\n"
        "            result.append(value)\n"
        "    return result\n",
        task,
        2.0,
    )
    assert passing["passed"]
    assert passing["tests_passed"] == passing["tests_total"] == 4

    failing = benchmark._score_source(
        "def stable_unique(values):\n    return sorted(set(values))\n", task, 2.0
    )
    assert not failing["passed"]
    assert failing["tests_passed"] < failing["tests_total"]


def test_deque_import_and_while_are_narrowly_allowed_and_executable() -> None:
    task = _task("grid_shortest_path")
    source = (
        "def shortest_grid_path(grid, start, goal):\n"
        "    from collections import deque\n"
        "    rows, cols = len(grid), len(grid[0])\n"
        "    queue = deque([(start[0], start[1], 0)])\n"
        "    visited = set([(start[0], start[1])])\n"
        "    while queue:\n"
        "        row, col, distance = queue.popleft()\n"
        "        if [row, col] == goal:\n"
        "            return distance\n"
        "        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:\n"
        "            nr, nc = row + dr, col + dc\n"
        "            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:\n"
        "                visited.add((nr, nc))\n"
        "                queue.append((nr, nc, distance + 1))\n"
        "    return -1\n"
    )
    score = benchmark._score_source(source, task, 2.0)
    assert score["passed"]
    assert score["tests_passed"] == score["tests_total"] == 4


def test_nonmutation_oracle_detects_shallow_nested_aliasing() -> None:
    task = _task("merge_intervals")
    shallow_aliasing_solution = (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals, key=lambda item: item[0])\n"
        "    merged = [ordered[0]]\n"
        "    for current in ordered[1:]:\n"
        "        if current[0] <= merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], current[1])\n"
        "        else:\n"
        "            merged.append(current)\n"
        "    return merged\n"
    )
    score = benchmark._score_source(shallow_aliasing_solution, task, 2.0)
    assert not score["passed"]
    assert any(
        test["exact_output_passed"] and not test["input_mutation_passed"]
        for test in score["tests"]
    )


def test_first_unique_oracle_uses_actual_first_unique_index() -> None:
    task = _task("first_unique_character")
    case = next(case for case in task["tests"] if case["args"] == ["112233!4!"])
    assert case["expected"] == 7


def test_stream_capture_preserves_exact_bytes_and_observes_first_output() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys,time; time.sleep(0.03); "
            "sys.stdout.write('  CODE\\n'); sys.stdout.flush(); "
            "sys.stderr.write('metric\\n'); sys.stderr.flush()"
        ),
    ]
    result = benchmark._stream_process(command, timeout_seconds=2.0)
    assert result.returncode == 0
    assert result.stdout.decode() == f"  CODE{os.linesep}"
    assert result.stderr.decode() == f"metric{os.linesep}"
    assert result.first_output_seconds is not None
    assert result.first_output_seconds >= 0.01
    assert not result.timed_out


def test_llama_timing_parser_distinguishes_load_decode_and_resident_estimate() -> None:
    stderr = """
load time = 12000.00 ms
prompt eval time = 200.00 ms / 100 tokens (500.00 tokens per second)
eval time = 800.00 ms / 80 runs (100.00 tokens per second)
total time = 13000.00 ms / 180 tokens
"""
    metrics = benchmark._llama_metrics(stderr)
    assert metrics["load_ms"] == 12000.0
    assert metrics["prompt_tokens"] == 100
    assert metrics["decode_runs"] == 80
    assert metrics["decode_tokens_per_second"] == 100.0
    assert metrics["resident_inference_ms_estimate"] == 1000.0
    assert metrics["resident_ttft_ms_estimate"] == 210.0


def test_aggregate_never_labels_subsequent_cli_processes_as_persistent_warm() -> None:
    def result(index: int, task_id: str, source_hash: str):
        return {
            "launch_index": index,
            "task_id": task_id,
            "passed": True,
            "extracted_source_sha256": source_hash,
            "wall_elapsed_seconds": 13.0,
            "time_to_first_non_whitespace_output_seconds": 12.5,
            "llama_metrics": {
                "load_ms": 12000.0,
                "resident_ttft_ms_estimate": 250.0,
                "resident_inference_ms_estimate": 900.0,
                "prompt_tokens_per_second": 450.0,
                "decode_tokens_per_second": 75.0,
            },
        }

    tasks = [_task("clamp_integer")]
    aggregate = benchmark._aggregate(
        [result(1, "clamp_integer", "abc"), result(2, "clamp_integer", "abc")],
        tasks,
    )
    assert aggregate["all_passed"]
    assert aggregate["task_summary"]["clamp_integer"]["exact_source_repeatable"]
    semantics = aggregate["subsequent_fresh_processes"]["semantics"]
    assert "fresh processes" in semantics
    assert "not persistent-model latency" in semantics


def test_hidden_evaluator_protocol_returns_json() -> None:
    task = _task("clamp_integer")
    payload = {
        "task": task,
        "source": (
            "def clamp(value, low, high):\n"
            "    return max(low, min(value, high))\n"
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--_evaluate"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    parsed = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert parsed["passed"]


def test_streamed_server_completion_preserves_sse_and_measures_ttft(monkeypatch) -> None:
    lines = [
        b'data: {"content":"def ","stop":false}\n',
        b"\n",
        (
            b'data: {"content":"clamp(value, low, high):\\n    return value",'
            b'"stop":true,"stop_type":"eos","timings":{"prompt_n":40,'
            b'"prompt_ms":100.0,"prompt_per_second":400.0,"predicted_n":12,'
            b'"predicted_ms":120.0,"predicted_per_second":100.0}}\n'
        ),
    ]
    observed = {}

    class FakeResponse:
        status = 200

        def __init__(self):
            self.headers = Message()
            self.headers["Content-Type"] = "text/event-stream"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(lines)

        def getcode(self):
            return self.status

    def fake_urlopen(request, timeout):
        observed["request"] = request
        observed["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(benchmark.urllib.request, "urlopen", fake_urlopen)
    result = benchmark._stream_server_completion(
        "http://127.0.0.1:18080",
        "<|im_start|>user\ncode<|im_end|>\n<|im_start|>assistant\n",
        max_new_tokens=64,
        seed=2026,
        timeout_seconds=5.0,
        cache_prompt=False,
    )
    request_body = json.loads(observed["request"].data)
    assert observed["request"].full_url.endswith("/completion")
    assert request_body["stream"] is True
    assert request_body["temperature"] == 0.0
    assert request_body["top_k"] == 1
    assert request_body["reasoning_format"] == "none"
    assert result.raw_response == b"".join(lines)
    assert result.generated_text.startswith("def clamp")
    assert result.first_output_seconds is not None
    assert result.error is None
    metrics = benchmark._server_metrics(result.events)
    assert metrics["prompt_tokens_per_second"] == 400.0
    assert metrics["decode_tokens_per_second"] == 100.0
    assert metrics["resident_inference_ms"] == 220.0


def test_server_preflight_pins_revision_model_context_and_gpu(
    monkeypatch, tmp_path
) -> None:
    artifact = tmp_path / "artifact"
    model = tmp_path / "models" / "model.gguf"
    artifact.mkdir()
    model.parent.mkdir()
    model.write_bytes(b"model")
    task_manifest = {
        "execution_profile": {
            "maximum_context_tokens": 8192,
            "sampling": {"temperature": 0.0, "top_k": 1, "seed": 2026},
        },
        "runtime": {"revision": "b6014", "build_number": 6014},
        "payload": {"path": "../models/model.gguf", "sha256": "model-sha"},
        "resource_contract": {"target_gpu": "Expected GPU"},
    }

    def fake_http(url, *, timeout_seconds):
        if url.endswith("/health"):
            return {"status": "ok"}
        return {
            "build_info": "b6014-b6014",
            "model_path": str(model.resolve()),
            "total_slots": 1,
            "default_generation_settings": {
                "n_ctx": 8192,
                "params": {"temperature": 0.0, "top_k": 1, "seed": 2026},
            },
        }

    monkeypatch.setattr(benchmark, "_http_json", fake_http)
    monkeypatch.setattr(
        benchmark,
        "_gpu_snapshot",
        lambda: {
            "name": "Expected GPU",
            "total_memory_mib": 12227,
            "used_memory_mib": 10000,
            "utilization_percent": 0,
        },
    )
    result = benchmark._server_preflight(
        "http://127.0.0.1:18080", artifact, task_manifest, 2048, 5.0
    )
    assert result["health"]["status"] == "ok"
    assert result["props"]["build_info"] == "b6014-b6014"
    assert result["binding"]["model_path"] == str(model.resolve())

    with pytest.raises(ValueError, match="127.0.0.1"):
        benchmark._server_preflight(
            "http://localhost:18080", artifact, task_manifest, 2048, 5.0
        )

    wrong_model = tmp_path / "other" / "model.gguf"
    wrong_model.parent.mkdir()
    wrong_model.write_bytes(b"different")

    def fake_wrong_model(url, *, timeout_seconds):
        response = fake_http(url, timeout_seconds=timeout_seconds)
        if url.endswith("/props"):
            response["model_path"] = str(wrong_model.resolve())
        return response

    monkeypatch.setattr(benchmark, "_http_json", fake_wrong_model)
    with pytest.raises(RuntimeError, match="server model mismatch"):
        benchmark._server_preflight(
            "http://127.0.0.1:18080", artifact, task_manifest, 2048, 5.0
        )


def _failed_merge_first_pass(session: Path):
    task = _task("merge_intervals")
    source = (
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    ordered = sorted(intervals, key=lambda item: item[0])\n"
        "    merged = [ordered[0]]\n"
        "    for current in ordered[1:]:\n"
        "        if current[0] <= merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], current[1])\n"
        "        else:\n"
        "            merged.append(current)\n"
        "    return merged\n"
    )
    score = benchmark._score_source(source, task, 2.0)
    assert not score["passed"]
    run_dir = session / "runs" / "001_r01_merge_intervals"
    run_dir.mkdir(parents=True)
    source_path = run_dir / "generated.py"
    source_path.write_text(source, encoding="utf-8")
    result = {
        "format": benchmark.FORMAT,
        "mode": "resident_server",
        "launch_index": 1,
        "repetition": 1,
        "task_id": task["id"],
        "passed": False,
        "generated_text_exact_after_runtime_marker_cleanup": source,
        "extracted_source_sha256": benchmark._sha256_bytes(source.encode()),
        "score": score,
        "raw_evidence": {"source_path": "runs/001_r01_merge_intervals/generated.py"},
        "wall_elapsed_seconds": 0.5,
        "time_to_first_non_whitespace_output_seconds": 0.1,
        "llama_metrics": {
            "load_ms": None,
            "resident_ttft_ms_estimate": 100.0,
            "resident_inference_ms_estimate": 500.0,
            "prompt_tokens_per_second": 400.0,
            "decode_tokens_per_second": 100.0,
        },
    }
    benchmark._write_json(run_dir / "result.json", result)
    return task, result, run_dir / "result.json"


def _correct_merge_source() -> str:
    return (
        "def merge_intervals(intervals):\n"
        "    ordered = sorted(intervals, key=lambda item: item[0])\n"
        "    merged = []\n"
        "    for start, end in ordered:\n"
        "        if merged and start <= merged[-1][1]:\n"
        "            merged[-1][1] = max(merged[-1][1], end)\n"
        "        else:\n"
        "            merged.append([start, end])\n"
        "    return merged\n"
    )


def test_repair_cli_defaults_to_disabled_and_zero_aggregate_is_unchanged() -> None:
    args = benchmark._build_parser().parse_args([])
    assert args.repair_attempts == 0
    first = {
        "launch_index": 1,
        "repetition": 1,
        "task_id": "clamp_integer",
        "passed": True,
        "extracted_source_sha256": "abc",
        "wall_elapsed_seconds": 1.0,
        "time_to_first_non_whitespace_output_seconds": 0.2,
        "llama_metrics": {
            "load_ms": None,
            "resident_ttft_ms_estimate": 200.0,
            "resident_inference_ms_estimate": 400.0,
            "prompt_tokens_per_second": 300.0,
            "decode_tokens_per_second": 90.0,
        },
    }
    aggregate = benchmark._aggregate([first], [_task("clamp_integer")])
    assert "first_pass_accuracy" not in aggregate
    assert "final_after_repair_accuracy" not in aggregate


def test_repair_feedback_reports_machine_mutation_facts(tmp_path) -> None:
    task, first, _ = _failed_merge_first_pass(tmp_path)
    facts = benchmark._repair_failure_facts(first)
    assert any("mutated argument 0" in fact for fact in facts)
    prompt = benchmark._repair_prompt(task, benchmark._first_source(tmp_path, first), facts)
    assert "Original requirement:" in prompt
    assert "Do not mutate intervals or any nested interval list" in prompt
    assert "mutated argument 0" in prompt
    assert "BEGIN PRIOR SOURCE" in prompt


def test_server_repair_is_separate_and_preserves_first_pass_verbatim(
    monkeypatch, tmp_path
) -> None:
    task, first, first_result_path = _failed_merge_first_pass(tmp_path)
    first_bytes = first_result_path.read_bytes()
    corrected = _correct_merge_source()
    raw = (
        b'data: {"content":"def merge_intervals(intervals):\\n","stop":false}\n\n'
        b'data: {"content":"    return []","stop":true,"stop_type":"eos",'
        b'"timings":{"prompt_n":200,"prompt_ms":400.0,"prompt_per_second":500.0,'
        b'"predicted_n":80,"predicted_ms":800.0,"predicted_per_second":100.0}}\n\n'
    )
    events = (
        {"content": corrected, "stop": False},
        {
            "content": "",
            "stop": True,
            "stop_type": "eos",
            "timings": {
                "prompt_n": 200,
                "prompt_ms": 400.0,
                "prompt_per_second": 500.0,
                "predicted_n": 80,
                "predicted_ms": 800.0,
                "predicted_per_second": 100.0,
            },
        },
    )
    observed = {}

    def fake_stream(server_url, prompt, **kwargs):
        observed["prompt"] = prompt
        observed["kwargs"] = kwargs
        return benchmark.ServerStreamResult(
            status_code=200,
            raw_response=raw,
            generated_text=corrected,
            events=events,
            response_headers={"content-type": "text/event-stream"},
            elapsed_seconds=1.25,
            first_output_seconds=0.22,
            error=None,
        )

    monkeypatch.setattr(benchmark, "_stream_server_completion", fake_stream)
    monkeypatch.setattr(benchmark, "_gpu_snapshot", lambda: None)
    monkeypatch.setattr(benchmark, "_sample_gpu", lambda stop, samples: None)
    repair = benchmark._run_repair_server(
        session=tmp_path,
        attempt=1,
        first_result=first,
        task=task,
        server_url="http://127.0.0.1:18080",
        max_new_tokens=384,
        seed=2026,
        process_timeout=30.0,
        evaluator_timeout=2.0,
        cache_prompt=False,
    )
    assert repair["passed"]
    assert repair["score"]["tests_passed"] == repair["score"]["tests_total"] == 4
    assert repair["time_to_first_non_whitespace_output_seconds"] == 0.22
    assert repair["llama_metrics"]["decode_tokens_per_second"] == 100.0
    assert any("mutated argument 0" in fact for fact in repair["machine_failure_facts"])
    assert first_result_path.read_bytes() == first_bytes
    assert repair["first_pass"]["result_sha256"] == benchmark._sha256_bytes(first_bytes)
    repair_dir = tmp_path / "repairs" / "001_r01_merge_intervals_a01"
    assert (repair_dir / "request.json").is_file()
    assert (repair_dir / "response.sse").read_bytes() == raw
    assert (repair_dir / "generated.py").read_text(encoding="utf-8").strip() == corrected.strip()
    assert "mutated argument 0" in observed["prompt"]


def test_cli_repair_path_records_independent_process_evidence(monkeypatch, tmp_path) -> None:
    task, first, first_result_path = _failed_merge_first_pass(tmp_path)
    first_bytes = first_result_path.read_bytes()
    corrected = _correct_merge_source()
    stderr = (
        "load time = 12000.00 ms\n"
        "prompt eval time = 400.00 ms / 200 tokens (500.00 tokens per second)\n"
        "eval time = 800.00 ms / 80 runs (100.00 tokens per second)\n"
        "total time = 13200.00 ms / 280 tokens\n"
    ).encode()
    monkeypatch.setattr(
        benchmark,
        "_stream_process",
        lambda command, timeout_seconds: benchmark.StreamResult(
            returncode=0,
            stdout=corrected.encode(),
            stderr=stderr,
            elapsed_seconds=13.4,
            first_output_seconds=12.6,
            timed_out=False,
        ),
    )
    monkeypatch.setattr(benchmark, "_gpu_snapshot", lambda: None)
    monkeypatch.setattr(benchmark, "_sample_gpu", lambda stop, samples: None)
    manifest = {
        "execution_profile": {
            "gpu_layers": 999,
            "kv_cache_k": "q8_0",
            "kv_cache_v": "q8_0",
        }
    }
    repair = benchmark._run_repair_cli(
        session=tmp_path,
        attempt=1,
        first_result=first,
        task=task,
        runtime=Path("llama-cli.exe"),
        model=Path("model.gguf"),
        manifest=manifest,
        context=2048,
        max_new_tokens=384,
        seed=2026,
        process_timeout=30.0,
        evaluator_timeout=2.0,
    )
    assert repair["passed"]
    assert repair["llama_metrics"]["load_ms"] == 12000.0
    assert repair["raw_evidence"]["stdout_sha256"] == benchmark._sha256_bytes(
        corrected.encode()
    )
    assert first_result_path.read_bytes() == first_bytes


def test_repair_aggregate_keeps_first_pass_failure_and_reports_final_pass() -> None:
    task = _task("merge_intervals")
    first = {
        "launch_index": 1,
        "repetition": 1,
        "task_id": task["id"],
        "passed": False,
        "extracted_source_sha256": "first",
        "wall_elapsed_seconds": 1.0,
        "time_to_first_non_whitespace_output_seconds": 0.2,
        "llama_metrics": {
            "load_ms": None,
            "resident_ttft_ms_estimate": 200.0,
            "resident_inference_ms_estimate": 600.0,
            "prompt_tokens_per_second": 400.0,
            "decode_tokens_per_second": 100.0,
        },
    }
    repair = {
        "parent_launch_index": 1,
        "parent_repetition": 1,
        "task_id": task["id"],
        "passed": True,
        "extracted_source_sha256": "repaired",
        "wall_elapsed_seconds": 0.8,
        "time_to_first_non_whitespace_output_seconds": 0.15,
        "llama_metrics": {"decode_tokens_per_second": 110.0},
    }
    first_aggregate = benchmark._aggregate([first], [task])
    combined = benchmark._aggregate_with_repairs(
        first_aggregate, [first], [repair], [task]
    )
    assert not combined["all_passed"]
    assert not combined["first_pass_accuracy"]["all_passed"]
    assert combined["final_after_repair_accuracy"]["all_passed"]
    assert combined["final_after_repair_all_passed"]
    assert combined["repair_attempts"]["executed"] == 1
    assert combined["repair_attempts"]["passed"] == 1
