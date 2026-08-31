from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import nhdf_edge.substrate_pdf as substrate_pdf
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
    assert sidecar["validation"]["body_heading_inventory"] == [
        "Referential closure",
        "Typed distinctions",
    ]
    assert sidecar["validation"]["body_sentinels"]
    determinism = sidecar["renderer"]["determinism"]
    assert determinism["cross_environment_guarantee"] is False
    assert "versions and hashes" in determinism["scope"]
    assert len(sidecar["renderer"]["fingerprint_sha256"]) == 64
    assert {font["role"] for font in sidecar["renderer"]["fonts"]} == {
        "normal",
        "bold",
        "italic",
        "bold_italic",
    }
    assert all(len(font["sha256"]) == 64 for font in sidecar["renderer"]["fonts"])
    assert {"python", "zlib", "reportlab", "pypdf"}.issubset(
        sidecar["renderer"]["dependencies"]
    )
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


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            '{"title":"First","title":"Second","version":"1","body":"x"}',
            "duplicate object key",
        ),
        (
            '{"title":"Strict JSON","version":"1","value":NaN}',
            "non-finite",
        ),
        (
            '{"title":"Strict JSON","version":"1","value":1e999}',
            "non-finite",
        ),
        (
            '{"title":"Strict JSON","version":"1","e\\u0301":1,"é":2}',
            "duplicate object key after NFC",
        ),
    ],
)
def test_json_source_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path, payload: str, match: str
) -> None:
    source = tmp_path / "strict.json"
    source.write_text(payload, encoding="utf-8")

    with pytest.raises(SourceFormatError, match=match):
        load_versioned_source(source)


def test_json_source_normalizes_nfc_and_negative_zero(tmp_path: Path) -> None:
    source = tmp_path / "normalized.json"
    source.write_text(
        '{"title":"Cafe\\u0301","version":"1","values":{"zero":-0.0}}',
        encoding="utf-8",
    )

    document = load_versioned_source(source)

    assert document.title == "Café"
    assert document.body["values"]["zero"] == 0.0
    assert math.copysign(1.0, document.body["values"]["zero"]) == 1.0


def test_automatic_body_validation_rejects_cover_only_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.md"
    output = tmp_path / "contract.pdf"
    _markdown_source(source)
    monkeypatch.setattr(substrate_pdf, "_markdown_flowables", lambda *_args: [])

    with pytest.raises(PdfValidationError, match="missing"):
        render_substrate_pdf(source, output)

    assert not output.exists()
    assert not substrate_pdf.metadata_sidecar_path(output).exists()


def test_sidecar_staging_failure_preserves_prior_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.md"
    output = tmp_path / "contract.pdf"
    _markdown_source(source)
    first = render_substrate_pdf(source, output)
    before = {
        output: output.read_bytes(),
        first.sidecar_path: first.sidecar_path.read_bytes(),
    }
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "This fixture makes no fixed-point convergence claim.",
            "This revised fixture still makes no fixed-point convergence claim.",
        ),
        encoding="utf-8",
        newline="\n",
    )

    def fail_sidecar(*_args, **_kwargs):
        raise RuntimeError("forced sidecar staging failure")

    monkeypatch.setattr(substrate_pdf, "_stage_json_file", fail_sidecar)
    with pytest.raises(RuntimeError, match="forced sidecar"):
        render_substrate_pdf(source, output)

    assert {path: path.read_bytes() for path in before} == before


def test_poppler_failure_preserves_prior_pdf_sidecar_and_page_set(tmp_path: Path) -> None:
    source = tmp_path / "contract.md"
    output = tmp_path / "contract.pdf"
    pages = tmp_path / "pages"
    pages.mkdir()
    _markdown_source(source)
    first = render_substrate_pdf(source, output)
    page_one = pages / "contract-page-001.png"
    page_extra = pages / "contract-page-002.png"
    page_one.write_bytes(b"prior-page-one")
    page_extra.write_bytes(b"prior-page-two")
    before = {
        output: output.read_bytes(),
        first.sidecar_path: first.sidecar_path.read_bytes(),
        page_one: page_one.read_bytes(),
        page_extra: page_extra.read_bytes(),
    }
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "Referential closure", "Revised referential closure"
        ),
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(substrate_pdf.PopplerUnavailableError):
        render_substrate_pdf(
            source,
            output,
            render_pages_to=pages,
            pdftoppm=tmp_path / "missing-pdftoppm",
        )

    assert {path: path.read_bytes() for path in before} == before


def test_page_publication_removes_stale_extra_images_transactionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "contract.md"
    output = tmp_path / "contract.pdf"
    pages = tmp_path / "pages"
    _markdown_source(source)
    render_substrate_pdf(source, output)
    pages.mkdir()
    page_one = pages / "contract-page-001.png"
    stale = pages / "contract-page-002.png"
    page_one.write_bytes(b"old-one")
    stale.write_bytes(b"old-two")

    def fake_run(arguments, **_kwargs):
        if arguments[1] == "-v":
            return SimpleNamespace(
                returncode=0,
                stderr="pdftoppm version test-double",
                stdout="",
            )
        prefix = Path(arguments[-1])
        prefix.with_name(prefix.name + "-1.png").write_bytes(
            b"\x89PNG\r\n\x1a\nnew"
        )
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(substrate_pdf.subprocess, "run", fake_run)
    rendered = render_substrate_pdf(
        source,
        output,
        render_pages_to=pages,
        pdftoppm=sys.executable,
    )

    assert rendered.rendered_pages == (page_one,)
    assert page_one.read_bytes() == b"\x89PNG\r\n\x1a\nnew"
    assert not stale.exists()
    sidecar = json.loads(rendered.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["renderer"]["dependencies"]["poppler"]["version"] == (
        "pdftoppm version test-double"
    )
    assert sidecar["visual_qa"]["pages"][0]["sha256"] == sha256_file(page_one)


def test_publication_replace_failure_rolls_back_every_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_pdf = tmp_path / "report.pdf"
    target_sidecar = tmp_path / "report.pdf.metadata.json"
    stale_page = tmp_path / "report-page-002.png"
    staged_pdf = tmp_path / "new-pdf.stage"
    staged_sidecar = tmp_path / "new-sidecar.stage"
    target_pdf.write_bytes(b"old-pdf")
    target_sidecar.write_bytes(b"old-sidecar")
    stale_page.write_bytes(b"old-page")
    staged_pdf.write_bytes(b"new-pdf")
    staged_sidecar.write_bytes(b"new-sidecar")
    real_replace = substrate_pdf.os.replace
    failed = False

    def fail_second_publish(source, target):
        nonlocal failed
        if Path(source) == staged_sidecar and not failed:
            failed = True
            raise OSError("forced replace failure")
        return real_replace(source, target)

    monkeypatch.setattr(substrate_pdf.os, "replace", fail_second_publish)
    with pytest.raises(OSError, match="forced replace"):
        substrate_pdf._publish_transaction(
            ((staged_pdf, target_pdf), (staged_sidecar, target_sidecar)),
            (stale_page,),
        )

    assert target_pdf.read_bytes() == b"old-pdf"
    assert target_sidecar.read_bytes() == b"old-sidecar"
    assert stale_page.read_bytes() == b"old-page"
