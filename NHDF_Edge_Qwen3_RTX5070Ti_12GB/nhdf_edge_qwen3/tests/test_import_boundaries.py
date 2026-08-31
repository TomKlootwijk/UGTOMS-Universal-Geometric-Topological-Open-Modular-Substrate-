from __future__ import annotations

import json
import subprocess
import sys


def _isolated_import(statement: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,sys,time; start=time.perf_counter(); "
                f"{statement}; "
                "print(json.dumps({'torch_loaded': 'torch' in sys.modules, "
                "'elapsed_seconds': time.perf_counter()-start}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


def test_package_import_does_not_load_torch() -> None:
    result = _isolated_import("import nhdf_edge")
    assert result["torch_loaded"] is False


def test_cli_import_does_not_load_torch() -> None:
    result = _isolated_import("import nhdf_edge.cli")
    assert result["torch_loaded"] is False


def test_public_tensor_api_remains_declared_for_lazy_loading() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,nhdf_edge; "
                "print(json.dumps({'exports': nhdf_edge.__all__, "
                "'version': nhdf_edge.__version__}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    result = json.loads(completed.stdout)
    assert result["version"] == "0.1.0"
    assert set(result["exports"]) == {
        "PackedTensor",
        "QuantizationPolicy",
        "quantize_tensor",
        "dequantize_tensor",
        "dequantize_rows",
    }
