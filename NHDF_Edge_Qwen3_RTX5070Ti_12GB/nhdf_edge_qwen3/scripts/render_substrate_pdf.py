#!/usr/bin/env python3
"""Render one local, versioned Markdown/JSON source as a verified PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from nhdf_edge.substrate_pdf import SubstratePdfError, render_substrate_pdf  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a local versioned Markdown/JSON substrate document, verify "
            "its expected text with pypdf, and write a SHA-256 metadata sidecar."
        )
    )
    parser.add_argument("source", help="UTF-8 .md, .markdown, or .json source")
    parser.add_argument(
        "--output",
        help="PDF destination (default: output/pdf/<source stem>.pdf)",
    )
    parser.add_argument("--title", help="override the source title")
    parser.add_argument("--version", help="override the source version")
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="TEXT",
        help="additional text that must be extractable from the PDF; repeatable",
    )
    parser.add_argument("--sidecar", help="override the .pdf.metadata.json path")
    parser.add_argument(
        "--render-pages",
        metavar="DIRECTORY",
        help="also render page PNGs with Poppler for visual QA",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Poppler DPI (72-600)")
    parser.add_argument(
        "--pdftoppm",
        default="pdftoppm",
        help="pdftoppm executable name or path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else PROJECT_ROOT / "output" / "pdf" / f"{source.stem}.pdf"
    )
    try:
        result = render_substrate_pdf(
            source,
            output,
            title=args.title,
            version=args.version,
            expected_text=args.expect,
            sidecar_path=args.sidecar,
            render_pages_to=args.render_pages,
            render_dpi=args.dpi,
            pdftoppm=args.pdftoppm,
        )
    except SubstratePdfError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "source": str(result.source_path),
                "source_sha256": result.source_sha256,
                "output": str(result.output_path),
                "output_sha256": result.output_sha256,
                "sidecar": str(result.sidecar_path),
                "pages": result.page_count,
                "rendered_pages": [str(path) for path in result.rendered_pages],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
