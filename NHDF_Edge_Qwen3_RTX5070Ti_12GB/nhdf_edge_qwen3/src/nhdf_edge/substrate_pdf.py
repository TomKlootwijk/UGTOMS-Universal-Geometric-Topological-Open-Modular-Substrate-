"""Versioned PDF publication pipeline with a recorded determinism scope.

The renderer accepts local Markdown or JSON only.  It never follows links,
loads remote assets, or reads an archive implicitly.  A build publishes a PDF,
optional Poppler pages, and a stable JSON sidecar as one rollback-capable file
set.  The sidecar records exact source/output hashes plus the renderer, font,
and dependency fingerprint within which byte repeatability is claimed.  Text
verification is performed by reopening the staged PDF with ``pypdf``.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
import zlib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SIDECAR_SCHEMA = "nhdf.substrate-pdf-build.v2"
RENDERER_ID = "nhdf-edge-substrate-pdf-1"
_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_UNORDERED = re.compile(r"^\s*[-*+]\s+(.+)$")
_ORDERED = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
_TABLE_DIVIDER = re.compile(r"^:?-{3,}:?$")
_FONT_SPECS = (
    ("normal", "NHDFSans", "Vera.ttf"),
    ("bold", "NHDFSans-Bold", "VeraBd.ttf"),
    ("italic", "NHDFSans-Italic", "VeraIt.ttf"),
    ("bold_italic", "NHDFSans-BoldItalic", "VeraBI.ttf"),
)
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


class SubstratePdfError(RuntimeError):
    """Base error for source, rendering, or validation failures."""


class SourceFormatError(SubstratePdfError):
    """The local source is unsupported, unversioned, or malformed."""


class PdfValidationError(SubstratePdfError):
    """The generated PDF could not be reopened or lacks required text."""


class PopplerUnavailableError(SubstratePdfError):
    """The requested visual-QA renderer is not installed or executable."""


@dataclass(frozen=True)
class SourceDocument:
    path: Path
    source_format: str
    title: str
    version: str
    body: Any
    metadata: Mapping[str, Any]
    source_sha256: str
    source_bytes: int


@dataclass(frozen=True)
class PdfValidationResult:
    page_count: int
    matched_text: tuple[str, ...]
    extracted_text_sha256: str


@dataclass(frozen=True)
class PdfBuildResult:
    source_path: Path
    output_path: Path
    sidecar_path: Path
    source_sha256: str
    output_sha256: str
    page_count: int
    rendered_pages: tuple[Path, ...] = ()


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_sidecar_path(pdf_path: str | os.PathLike[str]) -> Path:
    """Return the stable ``.pdf.metadata.json`` sidecar name."""

    path = Path(pdf_path)
    return path.with_suffix(path.suffix + ".metadata.json")


def _clean_text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise SourceFormatError(f"{name} must be a string")
    cleaned = unicodedata.normalize("NFC", value).translate(_DASH_TRANSLATION).strip()
    if required and not cleaned:
        raise SourceFormatError(f"{name} must be non-empty")
    return cleaned


def _freeze_json(value: Any, path: str = "$") -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SourceFormatError(f"{path} contains a non-string key")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in frozen:
                raise SourceFormatError(
                    f"{path} contains a duplicate key after NFC normalization: {normalized_key!r}"
                )
            frozen[normalized_key] = _freeze_json(
                item, f"{path}.{normalized_key}"
            )
        return MappingProxyType({key: frozen[key] for key in sorted(frozen)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SourceFormatError(f"{path} contains a non-finite number")
        return 0.0 if value == 0.0 else value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    raise SourceFormatError(f"{path} contains unsupported {type(value).__name__}")


def _strict_json_loads(value: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in result:
                raise ValueError(
                    f"duplicate object key after NFC normalization: {normalized_key!r}"
                )
            result[normalized_key] = item
        return result

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite numeric token {token!r} is not valid JSON")

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise SourceFormatError(f"invalid strict JSON source: {error}") from error
    if not isinstance(parsed, dict):
        raise SourceFormatError("JSON source root must be an object")
    normalized = _plain_json(_freeze_json(parsed))
    return dict(normalized)


def _plain_json(value: Any) -> Any:
    """Convert immutable internal JSON containers back to JSON primitives."""

    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _front_matter(markdown: str) -> tuple[dict[str, str], str]:
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise SourceFormatError("Markdown front matter has no closing --- marker")
    metadata: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines[1:closing], start=2):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise SourceFormatError(
                f"Markdown front matter line {line_number} must be key: value"
            )
        key, raw_value = line.split(":", 1)
        key = key.strip().lower().replace("-", "_")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise SourceFormatError(f"invalid front matter key {key!r}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key in metadata:
            raise SourceFormatError(f"duplicate front matter key {key!r}")
        metadata[key] = value
    return metadata, "\n".join(lines[closing + 1 :]).lstrip("\n")


def load_versioned_source(
    source_path: str | os.PathLike[str],
    *,
    title: str | None = None,
    version: str | None = None,
) -> SourceDocument:
    """Load one UTF-8 Markdown or JSON document and resolve title/version.

    Markdown may declare simple ``key: value`` front matter.  JSON may declare
    ``title`` and ``version`` at its root or in a root ``metadata`` object.
    Explicit arguments override those declarations, but a version must always
    be resolved before rendering.
    """

    path = Path(source_path).expanduser().resolve()
    if not path.is_file():
        raise SourceFormatError(f"source file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".md", ".markdown", ".json"}:
        raise SourceFormatError("source must use .md, .markdown, or .json")
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceFormatError("source must be valid UTF-8") from error

    metadata: dict[str, Any]
    body: Any
    if suffix in {".md", ".markdown"}:
        source_format = "markdown"
        metadata, body = _front_matter(decoded)
    else:
        source_format = "json"
        parsed = _strict_json_loads(decoded)
        raw_metadata = parsed.get("metadata", {})
        if raw_metadata is None:
            raw_metadata = {}
        if not isinstance(raw_metadata, dict):
            raise SourceFormatError("JSON metadata must be an object")
        metadata = dict(raw_metadata)
        for key in ("title", "version", "subtitle", "status", "author", "abstract"):
            if key in parsed:
                metadata[key] = parsed[key]
        body = {
            key: value
            for key, value in parsed.items()
            if key not in {"metadata", "title", "version", "subtitle", "status", "author", "abstract"}
        }

    resolved_title = title if title is not None else metadata.get("title")
    if resolved_title is None:
        resolved_title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    resolved_version = version if version is not None else metadata.get("version")
    if resolved_version is None:
        raise SourceFormatError(
            "source is unversioned; declare version in the source or pass version="
        )
    clean_title = _clean_text(resolved_title, "title")
    clean_version = _clean_text(resolved_version, "version")
    if _VERSION.fullmatch(clean_version) is None:
        raise SourceFormatError(
            "version must start with an alphanumeric character and contain only "
            "letters, digits, dots, underscores, plus signs, or hyphens"
        )
    metadata.pop("title", None)
    metadata.pop("version", None)
    return SourceDocument(
        path=path,
        source_format=source_format,
        title=clean_title,
        version=clean_version,
        body=_freeze_json(body) if source_format == "json" else body,
        metadata=_freeze_json(metadata),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
    )


def _safe_display_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).translate(_DASH_TRANSLATION)


def _inline_markup(value: str) -> str:
    """Escape source text and add a deliberately small inline-markup subset."""

    escaped = html.escape(_safe_display_text(value), quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_divider(line: str) -> bool:
    cells = _split_table_row(line)
    return bool(cells) and all(_TABLE_DIVIDER.fullmatch(cell) for cell in cells)


def _wrap_code(text: str, width: int = 92) -> str:
    lines: list[str] = []
    for raw_line in _safe_display_text(text).expandtabs(4).splitlines() or [""]:
        if len(raw_line) <= width:
            lines.append(raw_line)
            continue
        indentation = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        wrapped = textwrap.wrap(
            raw_line,
            width=width,
            subsequent_indent=indentation + "  ",
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return "\n".join(lines)


def _json_as_markdown(body: Mapping[str, Any]) -> str:
    """Convert a small report JSON shape, falling back to canonical JSON."""

    markdown = body.get("markdown")
    if isinstance(markdown, str):
        return markdown
    sections = body.get("sections")
    if not isinstance(sections, tuple):
        payload = _plain_json(body)
        return "## Source data\n\n```json\n" + json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n```"

    lines: list[str] = []
    for index, section in enumerate(sections, start=1):
        if not isinstance(section, Mapping):
            raise SourceFormatError(f"sections[{index - 1}] must be an object")
        heading = section.get("heading", section.get("title", f"Section {index}"))
        level = section.get("level", 2)
        if isinstance(level, bool) or not isinstance(level, int) or not 1 <= level <= 4:
            raise SourceFormatError(f"sections[{index - 1}].level must be 1 through 4")
        lines.extend(("#" * level + " " + _clean_text(heading, "section heading"), ""))
        content = section.get("markdown", section.get("text"))
        if content is not None:
            lines.extend((_clean_text(content, "section text", required=False), ""))
        bullets = section.get("bullets")
        if bullets is not None:
            if not isinstance(bullets, tuple):
                raise SourceFormatError(f"sections[{index - 1}].bullets must be an array")
            for item in bullets:
                lines.append("- " + _clean_text(item, "bullet"))
            lines.append("")
        code = section.get("code")
        if code is not None:
            lines.extend(("```", _clean_text(code, "code", required=False), "```", ""))
        table = section.get("table")
        if table is not None:
            if not isinstance(table, Mapping):
                raise SourceFormatError(f"sections[{index - 1}].table must be an object")
            headers = table.get("headers")
            rows = table.get("rows")
            if not isinstance(headers, tuple) or not headers:
                raise SourceFormatError("table headers must be a non-empty array")
            if not isinstance(rows, tuple):
                raise SourceFormatError("table rows must be an array")
            lines.append("| " + " | ".join(str(item) for item in headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
            for row in rows:
                if not isinstance(row, tuple) or len(row) != len(headers):
                    raise SourceFormatError("every table row must match the header width")
                lines.append("| " + " | ".join(str(item) for item in row) + " |")
            lines.append("")
    return "\n".join(lines)


def _document_markdown(document: SourceDocument) -> str:
    return (
        document.body
        if document.source_format == "markdown"
        else _json_as_markdown(document.body)
    )


def _visible_markdown_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"!\[([^]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    return " ".join(_safe_display_text(html.unescape(text)).split())


def _validation_excerpt(value: str, maximum: int = 120) -> str:
    if len(value) <= maximum:
        return value
    prefix = value[: maximum + 1]
    if " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.rstrip()


def _body_validation_inventory(
    document: SourceDocument,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive body-only validation evidence independent of caller hints."""

    markdown = _document_markdown(document)
    if not markdown.strip():
        raise SourceFormatError("document body must contain renderable content")
    headings: list[str] = []
    candidates: list[str] = []
    in_fence = False
    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if not stripped:
            continue
        heading = _HEADING.fullmatch(stripped)
        if heading is not None and not in_fence:
            visible = _visible_markdown_text(heading.group(2))
            if visible and visible not in headings:
                headings.append(visible)
            continue
        if not in_fence and _is_table_divider(stripped):
            continue
        unordered = _UNORDERED.fullmatch(stripped) if not in_fence else None
        ordered = _ORDERED.fullmatch(stripped) if not in_fence else None
        if unordered is not None:
            stripped = unordered.group(1)
        elif ordered is not None:
            stripped = ordered.group(2)
        elif stripped.startswith(">") and not in_fence:
            stripped = stripped[1:].lstrip()
        elif "|" in stripped and not in_fence:
            stripped = " ".join(_split_table_row(stripped))
        visible = _visible_markdown_text(stripped)
        if visible:
            candidates.append(_validation_excerpt(visible))

    cover_values = {
        _normalize_extracted_text(str(value))
        for value in (
            document.title,
            document.version,
            document.path.name,
            document.source_format,
            document.source_sha256,
            *document.metadata.values(),
        )
        if isinstance(value, (str, int, float))
    }
    sentinel = next(
        (
            item
            for item in (*candidates, *headings)
            if _normalize_extracted_text(item) not in cover_values
        ),
        None,
    )
    if sentinel is None:
        raise SourceFormatError(
            "document body needs text distinct from cover metadata for PDF validation"
        )
    return tuple(headings), (sentinel,)


def _font_records() -> tuple[dict[str, Any], ...]:
    """Resolve the exact ReportLab font assets included in the renderer scope."""

    import reportlab

    fonts_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    records: list[dict[str, Any]] = []
    for role, name, filename in _FONT_SPECS:
        font_path = fonts_dir / filename
        if not font_path.is_file():
            raise SubstratePdfError(
                f"required pinned ReportLab font is missing: {font_path}"
            )
        records.append(
            {
                "role": role,
                "postscript_name": name,
                "file": filename,
                "bytes": font_path.stat().st_size,
                "sha256": sha256_file(font_path),
            }
        )
    return tuple(records)


def _font_names() -> tuple[str, str, str, str]:
    """Register ReportLab's hash-recorded Vera family."""

    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts_dir = Path(reportlab.__file__).resolve().parent / "fonts"
    records = _font_records()
    names = tuple(str(record["postscript_name"]) for record in records)
    for record in records:
        name = str(record["postscript_name"])
        font_path = fonts_dir / str(record["file"])
        pdfmetrics.registerFont(TTFont(name, str(font_path)))
    pdfmetrics.registerFontFamily(
        "NHDFSans",
        normal=names[0],
        bold=names[1],
        italic=names[2],
        boldItalic=names[3],
    )
    return names


def _styles() -> tuple[Any, Mapping[str, Any]]:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    normal_font, bold_font, italic_font, _ = _font_names()
    palette = MappingProxyType(
        {
            "ink": colors.HexColor("#172337"),
            "muted": colors.HexColor("#5C6878"),
            "blue": colors.HexColor("#225D99"),
            "teal": colors.HexColor("#16858C"),
            "pale": colors.HexColor("#EDF4F8"),
            "line": colors.HexColor("#C9D7E2"),
            "code": colors.HexColor("#F3F5F7"),
            "white": colors.white,
        }
    )
    base = getSampleStyleSheet()
    styles = {
        "body": ParagraphStyle(
            "NHDFBody",
            parent=base["BodyText"],
            fontName=normal_font,
            fontSize=9.5,
            leading=14.2,
            textColor=palette["ink"],
            spaceAfter=7,
            alignment=TA_LEFT,
        ),
        "lead": ParagraphStyle(
            "NHDFLead",
            parent=base["BodyText"],
            fontName=normal_font,
            fontSize=11,
            leading=16,
            textColor=palette["muted"],
            spaceAfter=14,
        ),
        "title": ParagraphStyle(
            "NHDFTitle",
            parent=base["Title"],
            fontName=bold_font,
            fontSize=25,
            leading=30,
            textColor=palette["ink"],
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "NHDFH1",
            parent=base["Heading1"],
            fontName=bold_font,
            fontSize=17,
            leading=21,
            textColor=palette["blue"],
            spaceBefore=16,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "NHDFH2",
            parent=base["Heading2"],
            fontName=bold_font,
            fontSize=13,
            leading=17,
            textColor=palette["teal"],
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "NHDFH3",
            parent=base["Heading3"],
            fontName=bold_font,
            fontSize=11,
            leading=15,
            textColor=palette["ink"],
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h4": ParagraphStyle(
            "NHDFH4",
            parent=base["Heading4"],
            fontName=italic_font,
            fontSize=10,
            leading=14,
            textColor=palette["ink"],
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "NHDFBullet",
            parent=base["BodyText"],
            fontName=normal_font,
            fontSize=9.3,
            leading=13.5,
            leftIndent=14,
            firstLineIndent=-10,
            textColor=palette["ink"],
            spaceAfter=4,
        ),
        "quote": ParagraphStyle(
            "NHDFQuote",
            parent=base["BodyText"],
            fontName=italic_font,
            fontSize=9.5,
            leading=14,
            leftIndent=10,
            rightIndent=8,
            textColor=palette["muted"],
        ),
        "code": ParagraphStyle(
            "NHDFCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=10,
            leftIndent=7,
            rightIndent=7,
            borderPadding=7,
            borderColor=palette["line"],
            borderWidth=0.5,
            borderRadius=2,
            backColor=palette["code"],
            textColor=palette["ink"],
            spaceBefore=4,
            spaceAfter=9,
        ),
        "small": ParagraphStyle(
            "NHDFSmall",
            parent=base["BodyText"],
            fontName=normal_font,
            fontSize=7.4,
            leading=10,
            textColor=palette["muted"],
        ),
        "table_header": ParagraphStyle(
            "NHDFTableHeader",
            parent=base["BodyText"],
            fontName=bold_font,
            fontSize=7.8,
            leading=10.5,
            textColor=palette["white"],
        ),
    }
    return palette, MappingProxyType(styles)


def _markdown_flowables(markdown: str, styles: Mapping[str, Any], palette: Mapping[str, Any]) -> list[Any]:
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, Preformatted, Spacer, Table, TableStyle

    lines = markdown.splitlines()
    story: list[Any] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(_inline_markup(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            index += 1
            code_lines: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index >= len(lines):
                raise SourceFormatError("Markdown code fence is not closed")
            code = Preformatted(_wrap_code("\n".join(code_lines)), styles["code"])
            story.append(
                Table(
                    [[code]],
                    colWidths=(170 * mm,),
                    hAlign="LEFT",
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), palette["code"]),
                            ("BOX", (0, 0), (-1, -1), 0.5, palette["line"]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    ),
                )
            )
            story.append(Spacer(1, 8))
            index += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            story.append(
                Paragraph(_inline_markup(heading.group(2)), styles[f"h{len(heading.group(1))}"])
            )
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(lines)
            and "|" in lines[index + 1]
            and _is_table_divider(lines[index + 1])
        ):
            flush_paragraph()
            rows = [_split_table_row(line)]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_split_table_row(lines[index]))
                index += 1
            width = len(rows[0])
            if width == 0 or any(len(row) != width for row in rows):
                raise SourceFormatError("Markdown table rows must have equal width")
            cells = [
                [
                    Paragraph(
                        _inline_markup(cell),
                        styles["small"] if row_index else styles["table_header"],
                    )
                    for cell in row
                ]
                for row_index, row in enumerate(rows)
            ]
            table = Table(cells, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), palette["blue"]),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), _font_names()[1]),
                        ("GRID", (0, 0), (-1, -1), 0.4, palette["line"]),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, palette["pale"])),
                    ]
                )
            )
            story.extend((table, Spacer(1, 8)))
            continue
        unordered = _UNORDERED.match(line)
        ordered = _ORDERED.match(line)
        if unordered or ordered:
            flush_paragraph()
            marker = "-" if unordered else f"{ordered.group(1)}."
            content = unordered.group(1) if unordered else ordered.group(2)
            story.append(Paragraph(f"{marker} {_inline_markup(content)}", styles["bullet"]))
            index += 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            quote = stripped[1:].lstrip()
            story.append(
                Table(
                    [[Paragraph(_inline_markup(quote), styles["quote"]) ]],
                    style=TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), palette["pale"]),
                            ("LINEBEFORE", (0, 0), (0, -1), 3, palette["teal"]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 9),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 7),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ]
                    ),
                )
            )
            story.append(Spacer(1, 7))
            index += 1
            continue
        if stripped in {"---", "***", "___"}:
            flush_paragraph()
            story.extend(
                (
                    Spacer(1, 3),
                    HRFlowable(width="100%", thickness=0.6, color=palette["line"]),
                    Spacer(1, 5),
                )
            )
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def _build_pdf(document: SourceDocument, destination: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfgen import canvas as canvas_module
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    palette, styles = _styles()
    normal_font, bold_font, _, _ = _font_names()

    class InvariantCanvas(canvas_module.Canvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["invariant"] = 1
            kwargs["pageCompression"] = 1
            super().__init__(*args, **kwargs)

    doc = SimpleDocTemplate(
        str(destination),
        pagesize=A4,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=22 * mm,
        bottomMargin=19 * mm,
        title=document.title,
        author=str(document.metadata.get("author", "NHDF substrate project")),
        subject=f"Version {document.version}",
        creator=RENDERER_ID,
    )

    version_badge = Table(
        [[Paragraph(f"VERSION {html.escape(document.version)}", styles["small"]) ]],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), palette["pale"]),
                ("BOX", (0, 0), (-1, -1), 0.6, palette["teal"]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        ),
        hAlign="LEFT",
    )
    story: list[Any] = [
        Spacer(1, 12 * mm),
        version_badge,
        Spacer(1, 8 * mm),
        Paragraph(_inline_markup(document.title), styles["title"]),
    ]
    subtitle = document.metadata.get("subtitle")
    if isinstance(subtitle, str) and subtitle.strip():
        story.append(Paragraph(_inline_markup(subtitle), styles["lead"]))
    abstract = document.metadata.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        story.append(Paragraph(_inline_markup(abstract), styles["lead"]))
    story.extend(
        (
            Spacer(1, 4 * mm),
            HRFlowable(width="100%", thickness=1.1, color=palette["blue"]),
            Spacer(1, 5 * mm),
            Table(
                [
                    [Paragraph("Source", styles["small"]), Paragraph(html.escape(document.path.name), styles["small"])],
                    [Paragraph("Source SHA-256", styles["small"]), Paragraph(document.source_sha256, styles["small"])],
                    [Paragraph("Source format", styles["small"]), Paragraph(document.source_format, styles["small"])],
                ],
                colWidths=(34 * mm, 120 * mm),
                style=TableStyle(
                    [
                        ("FONTNAME", (0, 0), (0, -1), bold_font),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LINEBELOW", (0, 0), (-1, -2), 0.3, palette["line"]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 0),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                ),
            ),
            Spacer(1, 8 * mm),
        )
    )
    body_markdown = _document_markdown(document)
    story.extend(_markdown_flowables(body_markdown, styles, palette))

    def fit_text(value: str, maximum_width: float, font: str, size: float) -> str:
        candidate = value
        while candidate and pdfmetrics.stringWidth(candidate, font, size) > maximum_width:
            candidate = candidate[:-1]
        return candidate if candidate == value else candidate.rstrip() + "..."

    def decorate_page(pdf_canvas: Any, _doc: Any) -> None:
        width, height = A4
        pdf_canvas.saveState()
        pdf_canvas.setTitle(document.title)
        pdf_canvas.setAuthor(str(document.metadata.get("author", "NHDF substrate project")))
        pdf_canvas.setSubject(f"Version {document.version}")
        pdf_canvas.setCreator(RENDERER_ID)
        pdf_canvas.setStrokeColor(palette["line"])
        pdf_canvas.setLineWidth(0.5)
        pdf_canvas.line(19 * mm, height - 14 * mm, width - 19 * mm, height - 14 * mm)
        pdf_canvas.setFont(bold_font, 7.2)
        pdf_canvas.setFillColor(palette["blue"])
        header = fit_text(document.title, width - 75 * mm, bold_font, 7.2)
        pdf_canvas.drawString(19 * mm, height - 11.2 * mm, header)
        pdf_canvas.setFont(normal_font, 7.2)
        pdf_canvas.setFillColor(palette["muted"])
        pdf_canvas.drawRightString(
            width - 19 * mm,
            height - 11.2 * mm,
            f"Version {document.version}",
        )
        pdf_canvas.line(19 * mm, 13 * mm, width - 19 * mm, 13 * mm)
        pdf_canvas.setFont(normal_font, 6.6)
        pdf_canvas.drawString(19 * mm, 9.5 * mm, f"Source {document.source_sha256[:16]}")
        pdf_canvas.drawRightString(
            width - 19 * mm, 9.5 * mm, f"Page {pdf_canvas.getPageNumber()}"
        )
        pdf_canvas.restoreState()

    doc.build(
        story,
        onFirstPage=decorate_page,
        onLaterPages=decorate_page,
        canvasmaker=InvariantCanvas,
    )


def _normalize_extracted_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def validate_pdf_text(
    pdf_path: str | os.PathLike[str], expected_text: Iterable[str]
) -> PdfValidationResult:
    """Reopen a PDF with pypdf and require each expected text fragment."""

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise PdfValidationError("pypdf is required for PDF validation") from error

    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise PdfValidationError(f"PDF does not exist: {path}")
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise PdfValidationError("generated PDF is unexpectedly encrypted")
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    except PdfValidationError:
        raise
    except Exception as error:
        raise PdfValidationError(f"pypdf could not reopen {path.name}: {error}") from error
    if not reader.pages:
        raise PdfValidationError("generated PDF has no pages")
    normalized_document = _normalize_extracted_text(extracted)
    matched: list[str] = []
    missing: list[str] = []
    if isinstance(expected_text, str):
        raise PdfValidationError("expected_text must be an iterable of complete strings")
    for value in expected_text:
        expected = _clean_text(value, "expected text")
        if _normalize_extracted_text(expected) in normalized_document:
            matched.append(expected)
        else:
            missing.append(expected)
    if missing:
        raise PdfValidationError("PDF text validation failed; missing: " + ", ".join(repr(item) for item in missing))
    return PdfValidationResult(
        page_count=len(reader.pages),
        matched_text=tuple(matched),
        extracted_text_sha256=hashlib.sha256(extracted.encode("utf-8")).hexdigest(),
    )


def _resolve_executable(command: str | os.PathLike[str]) -> str:
    value = os.fspath(command)
    explicit = Path(value).expanduser()
    if explicit.parent != Path(".") or explicit.is_absolute():
        if explicit.is_file():
            return str(explicit.resolve())
        raise PopplerUnavailableError(f"Poppler executable does not exist: {explicit}")
    resolved = shutil.which(value)
    if resolved is None:
        raise PopplerUnavailableError(
            f"{value!r} was not found; install Poppler or pass its pdftoppm path"
        )
    return resolved


def _poppler_metadata(executable: str, timeout_seconds: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [executable, "-v"],
            capture_output=True,
            text=True,
            timeout=min(timeout_seconds, 15.0),
            check=False,
        )
        lines = [
            line.strip()
            for line in (
                (completed.stderr or "") + "\n" + (completed.stdout or "")
            ).splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError) as error:
        lines = [f"version probe failed: {type(error).__name__}"]
    path = Path(executable).resolve()
    return {
        "name": path.name,
        "version": lines[0] if lines else "unreported",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _render_pdf_pages_staged(
    pdf: Path,
    staging_directory: Path,
    *,
    dpi: int,
    pdftoppm: str | os.PathLike[str],
    timeout_seconds: float,
) -> tuple[tuple[Path, ...], dict[str, Any]]:
    executable = _resolve_executable(pdftoppm)
    metadata = _poppler_metadata(executable, timeout_seconds)
    prefix = staging_directory / "page"
    completed = subprocess.run(
        [executable, "-png", "-r", str(dpi), str(pdf), str(prefix)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise SubstratePdfError(
            f"pdftoppm failed with exit code {completed.returncode}: {detail}"
        )

    def page_number(path: Path) -> int:
        match = re.search(r"-(\d+)$", path.stem)
        if match is None:
            raise SubstratePdfError(f"unexpected pdftoppm page name: {path.name}")
        return int(match.group(1))

    generated = tuple(
        sorted(staging_directory.glob("page-*.png"), key=page_number)
    )
    if not generated:
        raise SubstratePdfError("pdftoppm produced no page images")
    actual_numbers = [page_number(path) for path in generated]
    if actual_numbers != list(range(1, len(generated) + 1)):
        raise SubstratePdfError(
            f"pdftoppm produced a non-contiguous page set: {actual_numbers!r}"
        )
    return generated, metadata


def _page_targets(pdf: Path, output: Path, count: int) -> tuple[Path, ...]:
    return tuple(
        output / f"{pdf.stem}-page-{number:03d}.png"
        for number in range(1, count + 1)
    )


def _published_page_files(pdf: Path, output: Path) -> tuple[Path, ...]:
    if not output.is_dir():
        return ()
    pattern = re.compile(rf"^{re.escape(pdf.stem)}-page-\d{{3,}}\.png$")
    return tuple(
        sorted(
            path
            for path in output.iterdir()
            if path.is_file() and pattern.fullmatch(path.name)
        )
    )


def _target_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _publish_transaction(
    replacements: Sequence[tuple[Path, Path]],
    removals: Iterable[Path] = (),
) -> None:
    """Publish a file set with process-local rollback on any replace failure."""

    replacement_rows = tuple((Path(source), Path(target)) for source, target in replacements)
    removal_paths = tuple(Path(path) for path in removals)
    target_keys: set[str] = set()
    for source, target in replacement_rows:
        if not source.is_file():
            raise SubstratePdfError(f"staged publication file is missing: {source}")
        key = _target_key(target)
        if key in target_keys:
            raise SubstratePdfError(f"duplicate publication target: {target}")
        target_keys.add(key)
        if not target.parent.is_dir():
            raise SubstratePdfError(f"publication directory does not exist: {target.parent}")
        if target.exists() and not target.is_file():
            raise SubstratePdfError(f"publication target is not a file: {target}")
    unique_removals: list[Path] = []
    for target in removal_paths:
        key = _target_key(target)
        if key in target_keys:
            continue
        target_keys.add(key)
        if target.exists() and not target.is_file():
            raise SubstratePdfError(f"removal target is not a file: {target}")
        unique_removals.append(target)

    affected = [target for _, target in replacement_rows]
    affected.extend(unique_removals)
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for target in affected:
            if not target.exists():
                continue
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.",
                suffix=".rollback",
                dir=target.parent,
                delete=False,
            ) as stream:
                backup = Path(stream.name)
            backup.unlink()
            os.replace(target, backup)
            backups.append((target, backup))
        for source, target in replacement_rows:
            os.replace(source, target)
            published.append(target)
    except Exception as error:
        rollback_errors: list[str] = []
        for target in reversed(published):
            try:
                target.unlink(missing_ok=True)
            except OSError as rollback_error:
                rollback_errors.append(f"remove {target}: {rollback_error}")
        for target, backup in reversed(backups):
            try:
                if backup.exists():
                    os.replace(backup, target)
            except OSError as rollback_error:
                rollback_errors.append(f"restore {target}: {rollback_error}")
        if rollback_errors:
            raise SubstratePdfError(
                "publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from error
        raise
    else:
        for _, backup in backups:
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass


def render_pdf_pages(
    pdf_path: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    *,
    dpi: int = 150,
    pdftoppm: str | os.PathLike[str] = "pdftoppm",
    timeout_seconds: float = 120.0,
) -> tuple[Path, ...]:
    """Render every PDF page to stable PNG names using Poppler."""

    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 72 <= dpi <= 600:
        raise SubstratePdfError("dpi must be an integer from 72 through 600")
    if timeout_seconds <= 0:
        raise SubstratePdfError("timeout_seconds must be positive")
    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.is_file():
        raise SubstratePdfError(f"PDF does not exist: {pdf}")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nhdf-pdf-pages-", dir=output) as temporary:
        generated, _ = _render_pdf_pages_staged(
            pdf,
            Path(temporary),
            dpi=dpi,
            pdftoppm=pdftoppm,
            timeout_seconds=timeout_seconds,
        )
        stable = _page_targets(pdf, output, len(generated))
        stale = set(_published_page_files(pdf, output)) - set(stable)
        _publish_transaction(tuple(zip(generated, stable)), stale)
    return stable


def _serialized_json(payload: Mapping[str, Any]) -> str:
    normalized = _plain_json(_freeze_json(payload))
    return json.dumps(
        normalized,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _stage_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialized_json(payload)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".json.tmp",
            prefix=path.name + ".",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = _stage_json_file(path, payload)
    try:
        _publish_transaction(((temporary, path),))
    finally:
        temporary.unlink(missing_ok=True)


def _module_dependency(module: Any, version: str) -> dict[str, Any]:
    module_path = Path(module.__file__).resolve()
    return {
        "version": version,
        "module_file": module_path.name,
        "module_sha256": sha256_file(module_path),
    }


def _renderer_manifest(poppler: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        import pypdf
        import reportlab
    except ImportError as error:  # pragma: no cover - imports are required earlier
        raise SubstratePdfError("reportlab and pypdf are required") from error
    executable = Path(sys.executable).resolve()
    dependencies: dict[str, Any] = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "byteorder": sys.byteorder,
            "executable_name": executable.name,
            "executable_sha256": sha256_file(executable),
        },
        "zlib": {"version": zlib.ZLIB_VERSION},
        "reportlab": _module_dependency(reportlab, str(reportlab.Version)),
        "pypdf": _module_dependency(pypdf, str(pypdf.__version__)),
    }
    if poppler is not None:
        dependencies["poppler"] = dict(poppler)
    manifest: dict[str, Any] = {
        "id": RENDERER_ID,
        "determinism": {
            "scope": (
                "identical source bytes, render options, renderer source, dependency "
                "versions and hashes, platform, and font hashes"
            ),
            "cross_environment_guarantee": False,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "renderer_source": {
            "name": Path(__file__).name,
            "sha256": sha256_file(__file__),
        },
        "dependencies": dependencies,
        "fonts": list(_font_records()),
    }
    fingerprint_bytes = json.dumps(
        _plain_json(_freeze_json(manifest)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest["fingerprint_sha256"] = hashlib.sha256(fingerprint_bytes).hexdigest()
    return manifest


def render_substrate_pdf(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    title: str | None = None,
    version: str | None = None,
    expected_text: Iterable[str] = (),
    sidecar_path: str | os.PathLike[str] | None = None,
    render_pages_to: str | os.PathLike[str] | None = None,
    render_dpi: int = 150,
    pdftoppm: str | os.PathLike[str] = "pdftoppm",
) -> PdfBuildResult:
    """Build, reopen, validate, hash, and describe one substrate PDF."""

    document = load_versioned_source(source_path, title=title, version=version)
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".pdf":
        raise SubstratePdfError("output path must end in .pdf")
    if output == document.path:
        raise SubstratePdfError("output PDF must not overwrite its source")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = (
        Path(sidecar_path).expanduser().resolve()
        if sidecar_path is not None
        else metadata_sidecar_path(output)
    )
    if sidecar in {document.path, output}:
        raise SubstratePdfError("sidecar path must be distinct from source and PDF")

    if isinstance(expected_text, str):
        raise SubstratePdfError("expected_text must be an iterable of complete strings")
    requested = tuple(expected_text)
    body_headings, body_sentinels = _body_validation_inventory(document)
    validation_terms = tuple(
        dict.fromkeys(
            (
                document.title,
                document.version,
                *body_headings,
                *body_sentinels,
                *requested,
            )
        )
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    page_output = (
        Path(render_pages_to).expanduser().resolve()
        if render_pages_to is not None
        else None
    )
    if page_output is not None:
        page_output.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf.tmp",
            prefix=output.stem + ".",
            dir=output.parent,
            delete=False,
        ) as stream:
            staged_pdf = Path(stream.name)
        stack.callback(staged_pdf.unlink, missing_ok=True)
        _build_pdf(document, staged_pdf)
        validation = validate_pdf_text(staged_pdf, validation_terms)
        output_digest = sha256_file(staged_pdf)

        generated_pages: tuple[Path, ...] = ()
        rendered_pages: tuple[Path, ...] = ()
        stale_pages: tuple[Path, ...] = ()
        poppler_metadata: dict[str, Any] | None = None
        if page_output is not None:
            page_staging = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(
                        prefix="nhdf-pdf-pages-", dir=page_output
                    )
                )
            )
            generated_pages, poppler_metadata = _render_pdf_pages_staged(
                staged_pdf,
                page_staging,
                dpi=render_dpi,
                pdftoppm=pdftoppm,
                timeout_seconds=120.0,
            )
            rendered_pages = _page_targets(
                output, page_output, len(generated_pages)
            )
            stale_pages = tuple(
                set(_published_page_files(output, page_output))
                - set(rendered_pages)
            )

        renderer = _renderer_manifest(poppler_metadata)
        sidecar_payload = {
            "schema": SIDECAR_SCHEMA,
            "document": {
                "title": document.title,
                "version": document.version,
                "metadata": document.metadata,
            },
            "source": {
                "name": document.path.name,
                "format": document.source_format,
                "bytes": document.source_bytes,
                "sha256": document.source_sha256,
            },
            "output": {
                "name": output.name,
                "bytes": staged_pdf.stat().st_size,
                "pages": validation.page_count,
                "sha256": output_digest,
            },
            "validation": {
                "method": "pypdf-text-extraction",
                "expected_text": validation.matched_text,
                "body_heading_inventory": body_headings,
                "body_sentinels": body_sentinels,
                "extracted_text_sha256": validation.extracted_text_sha256,
                "passed": True,
            },
            "renderer": renderer,
            "visual_qa": {
                "renderer": "pdftoppm" if rendered_pages else None,
                "renderer_fingerprint_sha256": (
                    renderer["fingerprint_sha256"] if rendered_pages else None
                ),
                "dpi": render_dpi if rendered_pages else None,
                "pages": [
                    {"name": target.name, "sha256": sha256_file(staged)}
                    for staged, target in zip(generated_pages, rendered_pages)
                ],
            },
        }
        staged_sidecar = _stage_json_file(sidecar, sidecar_payload)
        stack.callback(staged_sidecar.unlink, missing_ok=True)
        replacements = [
            (staged_pdf, output),
            (staged_sidecar, sidecar),
            *zip(generated_pages, rendered_pages),
        ]
        _publish_transaction(replacements, stale_pages)

    return PdfBuildResult(
        source_path=document.path,
        output_path=output,
        sidecar_path=sidecar,
        source_sha256=document.source_sha256,
        output_sha256=output_digest,
        page_count=validation.page_count,
        rendered_pages=tuple(rendered_pages),
    )


__all__ = [
    "PdfBuildResult",
    "PdfValidationError",
    "PdfValidationResult",
    "PopplerUnavailableError",
    "SIDECAR_SCHEMA",
    "SourceDocument",
    "SourceFormatError",
    "SubstratePdfError",
    "load_versioned_source",
    "metadata_sidecar_path",
    "render_pdf_pages",
    "render_substrate_pdf",
    "sha256_file",
    "validate_pdf_text",
]
