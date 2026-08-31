from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from nhdf_edge.substrate_pdf import (
    PdfValidationError,
    SIDECAR_SCHEMA,
    SourceFormatError,
    load_versioned_source,
    render_pdf_pages,
    render_substrate_pdf,
    sha256_file,
    validate_pdf_text,
)


def _markdown_source(path: Path) -> None:
    path.write_text(
        """---
title: Typed Substrate Contract
version: 0.4-test
subtitle: Deterministic report pipeline fixture
status: test-only
---

# Referential closure

The observable at generation `n` may update the residual at generation **n + 1**.

## Typed distinctions

- Cone slant length is separate from linear time.
- A radix trie is separate from comparison BST ordering.

| Role | Symbol |
| --- | --- |
| Linear time | time |
| Modular tick | X |

> This fixture makes no fixed-point convergence claim.

```text
observable[n] -> residual[n + 1]
```
""",
        encoding="utf-8",
        newline="\n",
    )


def test_markdown_pdf_is_deterministic_validated_and_hashed(tmp_path: Path) -> None:
    source = tmp_path / "contract.md"
    output = tmp_path / "contract.pdf"
    _markdown_source(source)

    first = render_substrate_pdf(
        source,
        output,
        expected_text=("Referential closure", "no fixed-point convergence claim"),
    )
    first_bytes = output.read_bytes()
    first_sidecar = first.sidecar_path.read_bytes()
    second = render_substrate_pdf(
        source,
        output,
        expected_text=("Referential closure", "no fixed-point convergence claim"),
    )

    assert output.read_bytes() == first_bytes
    assert second.sidecar_path.read_bytes() == first_sidecar
    assert first.output_sha256 == second.output_sha256 == sha256_file(output)
    assert first.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert first.page_count >= 1

    sidecar = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["schema"] == SIDECAR_SCHEMA
    assert sidecar["source"]["sha256"] == first.source_sha256
    assert sidecar["output"]["sha256"] == first.output_sha256
    assert sidecar["output"]["pages"] == first.page_count
    assert sidecar["validation"]["passed"] is True
    assert sidecar["renderer"]["deterministic"] is True
    assert sidecar["visual_qa"]["pages"] == []

    validation = validate_pdf_text(
        output,
        ("Typed Substrate Contract", "0.4-test", "radix trie", "comparison BST"),
    )
    assert validation.page_count == first.page_count


def test_versioned_json_sections_render_as_document_content(tmp_path: Path) -> None:
    source = tmp_path / "profile.json"
    source.write_text(
        json.dumps(
            {
                "title": "SCLP Profile Fixture",
                "version": "3.6.2-test",
                "metadata": {"status": "test-only"},
                "sections": [
                    {
                        "heading": "Symbol firewall",
                        "level": 1,
                        "text": "Each one-bit role remains independently typed.",
                        "bullets": ["payload parity", "topology orientation", "jitter", "branch predicate"],
                        "table": {
                            "headers": ["Geometry", "Meaning"],
                            "rows": [
                                ["implicit cone", "relation field"],
                                ["finite cone SDF", "exact distance"],
                                ["sweep interval", "certified bound"],
                            ],
                        },
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = render_substrate_pdf(
        source,
        tmp_path / "profile.pdf",
        expected_text=("Symbol firewall", "finite cone SDF", "sweep interval"),
    )
    assert result.page_count >= 1
    assert json.loads(result.sidecar_path.read_text(encoding="utf-8"))["source"]["format"] == "json"


def test_arbitrary_versioned_json_fallback_preserves_nested_values(tmp_path: Path) -> None:
    source = tmp_path / "contract.json"
    source.write_text(
        json.dumps(
            {
                "title": "Raw Contract Fixture",
                "version": "1.0-test",
                "contract": {
                    "closure": "next-generation-only",
                    "limits": {"maximum_steps": 32},
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result = render_substrate_pdf(
        source,
        tmp_path / "contract.pdf",
        expected_text=("next-generation-only", "maximum_steps"),
    )
    assert result.page_count >= 1


def test_source_must_resolve_a_version_and_supported_format(tmp_path: Path) -> None:
    unversioned = tmp_path / "unversioned.md"
    unversioned.write_text("# No version\n", encoding="utf-8")
    with pytest.raises(SourceFormatError, match="unversioned"):
        load_versioned_source(unversioned)
    loaded = load_versioned_source(unversioned, version="1.0")
    assert loaded.version == "1.0"

    unsupported = tmp_path / "source.txt"
    unsupported.write_text("version: 1", encoding="utf-8")
    with pytest.raises(SourceFormatError, match=".md"):
        load_versioned_source(unsupported)


def test_pypdf_validation_reports_missing_text(tmp_path: Path) -> None:
    source = tmp_path / "contract.md"
    _markdown_source(source)
    result = render_substrate_pdf(source, tmp_path / "contract.pdf")
    with pytest.raises(PdfValidationError, match="missing"):
        validate_pdf_text(result.output_path, ("this sentence is deliberately absent",))


def test_poppler_renders_stable_page_pngs_for_visual_qa(tmp_path: Path) -> None:
    executable = shutil.which("pdftoppm")
    if executable is None:
        pytest.skip("Poppler is not installed in this test environment")
    source = tmp_path / "contract.md"
    _markdown_source(source)
    result = render_substrate_pdf(source, tmp_path / "contract.pdf")
    pages = render_pdf_pages(
        result.output_path,
        tmp_path / "pages",
        dpi=96,
        pdftoppm=executable,
    )
    assert len(pages) == result.page_count
    assert [path.name for path in pages] == [
        f"contract-page-{number:03d}.png" for number in range(1, len(pages) + 1)
    ]
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in pages)
