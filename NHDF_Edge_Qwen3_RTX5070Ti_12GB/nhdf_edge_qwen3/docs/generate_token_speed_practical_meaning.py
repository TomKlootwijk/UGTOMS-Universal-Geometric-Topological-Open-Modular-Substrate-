from __future__ import annotations

import json
import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZATION_EVIDENCE = ROOT / "metrics" / "local" / "runtime_optimization_20260831.json"
GATE_EVIDENCE = (
    ROOT
    / "packs"
    / "qwen3-30b-a3b-nhdf-v03-iq2m"
    / "evidence"
    / "functional_gate.json"
)
OUTPUT = ROOT / "output" / "pdf" / "NHDF_Token_Speed_Practical_Meaning.pdf"


NAVY = HexColor("#102744")
NAVY_2 = HexColor("#173A68")
BLUE = HexColor("#2768E8")
BLUE_PALE = HexColor("#EAF1FF")
TEAL = HexColor("#128D93")
TEAL_PALE = HexColor("#E4F5F2")
GOLD = HexColor("#D87800")
GOLD_PALE = HexColor("#FFF3DB")
INK = HexColor("#172236")
MUTED = HexColor("#56657A")
LINE = HexColor("#D4DEEB")
PAGE = HexColor("#F5F8FC")
WHITE = HexColor("#FFFFFF")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fit_lines(text: str, font: str, size: float, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or stringWidth(candidate, font, size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: float = 8.4,
    leading: float = 11.2,
    color=INK,
) -> float:
    pdf.setFont(font, size)
    pdf.setFillColor(color)
    for line in _fit_lines(text, font, size, width):
        pdf.drawString(x, y, line)
        y -= leading
    return y


def _label(pdf: canvas.Canvas, text: str, x: float, y: float, color=BLUE) -> None:
    pdf.setFillColor(color)
    pdf.setFont("Helvetica-Bold", 7.3)
    pdf.drawString(x, y, text.upper())


def _card(pdf: canvas.Canvas, x: float, y: float, w: float, h: float, fill=WHITE) -> None:
    pdf.setFillColor(fill)
    pdf.setStrokeColor(LINE)
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, w, h, 12, fill=1, stroke=1)


def _bullet(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    dot=TEAL,
    size: float = 8.0,
) -> float:
    pdf.setFillColor(dot)
    pdf.circle(x + 2.5, y + 2.4, 2.3, fill=1, stroke=0)
    return _paragraph(pdf, text, x + 12, y + 5, width - 12, size=size, leading=10.5)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f} s"


def build() -> Path:
    optimization = _load(OPTIMIZATION_EVIDENCE)
    gate = _load(GATE_EVIDENCE)
    resident = optimization["resident_server"]
    model = optimization["model"]
    controlled = optimization["controlled_microbenchmark"]["current_winner"]
    aggregate = gate["aggregate"]

    coding_rate = float(resident["median_generation_tokens_per_second"])
    ttft_seconds = float(resident["median_warm_cached_ttft_ms"]) / 1000.0
    response_sizes = [64, 128, 256, 512]
    response_times = {
        tokens: ttft_seconds + (tokens - 1) / coding_rate for tokens in response_sizes
    }
    request_wall = resident["coding_request_wall_seconds"]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=A4)
    width, height = A4
    pdf.setTitle("NHDF Token Speed: Practical Meaning")
    pdf.setSubject(
        "Measured practical interpretation of local Qwen3-30B-A3B coding throughput"
    )
    pdf.setAuthor("UGTOMS / NHDF Edge validation")
    pdf.setCreator("NHDF evidence report generator")

    pdf.setFillColor(PAGE)
    pdf.rect(0, 0, width, height, fill=1, stroke=0)

    # Header
    pdf.setFillColor(NAVY)
    pdf.rect(0, height - 137, width, 137, fill=1, stroke=0)
    _label(pdf, "NHDF EDGE - MEASURED PRACTICAL INTERPRETATION", 38, height - 31, HexColor("#9EC1FF"))
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(
        38,
        height - 67,
        f"What {coding_rate:.2f} tokens/s means in local coding",
    )
    pdf.setFont("Helvetica", 10)
    pdf.setFillColor(HexColor("#D7E2F2"))
    pdf.drawString(
        38,
        height - 90,
        "Qwen3-30B-A3B on this RTX 5070 Ti Laptop GPU (12 GB), kept resident",
    )
    pdf.setFillColor(NAVY_2)
    pdf.roundRect(38, height - 122, 326, 20, 10, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 7.6)
    pdf.drawString(48, height - 115, "MEASURED ON THIS LAPTOP - NOT A GUARANTEE FOR EVERY PROMPT")

    margin = 38
    gutter = 14
    col_w = (width - 2 * margin - gutter) / 2

    # Main measured result
    top_y = height - 286
    _card(pdf, margin, top_y, col_w, 128, WHITE)
    _label(pdf, "Resident coding measurement", margin + 17, top_y + 105)
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 31)
    pdf.drawString(margin + 17, top_y + 69, f"{coding_rate:.2f}")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(margin + 142, top_y + 76, "tokens/s median")
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(
        margin + 17,
        top_y + 40,
        f"{resident['median_warm_cached_ttft_ms']:.2f} ms",
    )
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8.1)
    pdf.drawString(margin + 99, top_y + 43, "median warm cached time to first text")
    pdf.setStrokeColor(LINE)
    pdf.line(margin + 17, top_y + 28, margin + col_w - 17, top_y + 28)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 7.5)
    pdf.drawString(
        margin + 17,
        top_y + 13,
        f"Controlled 256-token average: {controlled['generation_average_tokens_per_second']:.2f} tok/s",
    )

    # Human-scale translation
    right_x = margin + col_w + gutter
    _card(pdf, right_x, top_y, col_w, 128, TEAL_PALE)
    _label(pdf, "Idealized warm output time", right_x + 17, top_y + 105, TEAL)
    pdf.setFont("Helvetica-Bold", 9.2)
    pdf.setFillColor(INK)
    rows = [(64, 78), (128, 56), (256, 34), (512, 12)]
    for tokens, y_offset in rows:
        pdf.drawString(right_x + 18, top_y + y_offset, f"{tokens:>3} tokens")
        pdf.setFillColor(TEAL)
        pdf.drawRightString(
            right_x + col_w - 18,
            top_y + y_offset,
            _format_seconds(response_times[tokens]),
        )
        pdf.setFillColor(INK)

    # Practical-use strip
    mid_y = height - 407
    _card(pdf, margin, mid_y, width - 2 * margin, 103, BLUE_PALE)
    _label(pdf, "What this feels like", margin + 17, mid_y + 82)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12.2)
    pdf.drawString(
        margin + 17,
        mid_y + 57,
        "Short coding answers arrive in roughly half a second;",
    )
    pdf.drawString(
        margin + 17,
        mid_y + 41,
        "long snippets take a few seconds.",
    )
    _paragraph(
        pdf,
        (
            "Across the measured coding requests, 37-225 output tokens took "
            f"{request_wall['minimum']:.3f}-{request_wall['maximum']:.3f} s, "
            f"with a {request_wall['median']:.3f} s median. At this speed, reading, "
            "testing, and correcting code usually take longer than generation itself."
        ),
        margin + 17,
        mid_y + 22,
        width - 2 * margin - 34,
        size=7.8,
        leading=9.5,
        color=MUTED,
    )

    # Quality and accessibility cards
    lower_y = height - 571
    _card(pdf, margin, lower_y, col_w, 154, WHITE)
    _label(pdf, "Speed with measured coding correctness", margin + 17, lower_y + 131)
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(margin + 17, lower_y + 98, "5/6")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 9.4)
    pdf.drawString(margin + 73, lower_y + 104, "task types passed both first runs")
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 23)
    pdf.drawString(margin + 17, lower_y + 64, "6/6")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 9.4)
    pdf.drawString(margin + 73, lower_y + 70, "passed after one machine-feedback repair")
    _paragraph(
        pdf,
        f"The repeated failure mutated nested input lists. One repair corrected both runs in {resident['median_repair_wall_seconds']:.2f} s median. This is tool-assisted quality, not perfect first-shot accuracy.",
        margin + 17,
        lower_y + 43,
        col_w - 34,
        size=7.8,
        leading=9.7,
        color=MUTED,
    )

    _card(pdf, right_x, lower_y, col_w, 154, GOLD_PALE)
    _label(pdf, "Literal access benefit", right_x + 17, lower_y + 131, GOLD)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 13.5)
    pdf.drawString(right_x + 17, lower_y + 105, "~58.2 GiB BF16 weights")
    pdf.setFont("Helvetica", 8.3)
    pdf.setFillColor(MUTED)
    pdf.drawString(right_x + 17, lower_y + 89, "would need about 4.76x this GPU's total VRAM")
    y = lower_y + 66
    y = _bullet(
        pdf,
        f"Complete {model['parameters'] / 1e9:.3f}B-parameter MoE model; 49/49 layers on GPU.",
        right_x + 17,
        y,
        col_w - 34,
        dot=GOLD,
    )
    y = _bullet(
        pdf,
        f"8K gate peak: {aggregate['peak_gpu_memory_mib']:,} MiB; {aggregate['headroom_mib']:,} MiB headroom.",
        right_x + 17,
        y - 3,
        col_w - 34,
        dot=GOLD,
    )
    _bullet(
        pdf,
        "External GGUF/IQ2_M carries the low-bit weights; NHDF seals and operates the deployment.",
        right_x + 17,
        y - 18,
        col_w - 34,
        dot=GOLD,
        size=7.5,
    )

    # Bottom conclusion
    conclusion_y = 148
    pdf.setFillColor(NAVY)
    pdf.roundRect(margin, conclusion_y, width - 2 * margin, 79, 13, fill=1, stroke=0)
    _label(pdf, "Practical conclusion", margin + 18, conclusion_y + 59, HexColor("#9EC1FF"))
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 11.6)
    pdf.drawString(
        margin + 18,
        conclusion_y + 37,
        "Generation is fast enough to feel immediate;",
    )
    pdf.drawString(
        margin + 18,
        conclusion_y + 23,
        "correctness checks are now the main delay.",
    )
    pdf.setFont("Helvetica", 8.1)
    pdf.setFillColor(HexColor("#D7E2F2"))
    pdf.drawString(
        margin + 18,
        conclusion_y + 8,
        f"Keep the server resident to avoid the {resident['startup_to_listening_seconds']:.2f} s cold startup. One configured slot means concurrent requests queue.",
    )

    # Caveats and provenance
    _label(pdf, "Boundaries of this fact", margin, 137, MUTED)
    caveat = (
        "The warm timing used prefix caching. Code has no reliable words-per-token conversion. "
        "The accuracy check is six small deterministic Python tasks, not repository-scale coding. "
        "Long contexts, thermals, and power state can change throughput; Windows was on Balanced."
    )
    _paragraph(pdf, caveat, margin, 122, width - 2 * margin, size=7.2, leading=9.2, color=MUTED)
    pdf.setStrokeColor(LINE)
    pdf.line(margin, 71, width - margin, 71)
    pdf.setFont("Helvetica", 6.4)
    pdf.setFillColor(MUTED)
    pdf.drawString(margin, 57, "Evidence: metrics/local/runtime_optimization_20260831.json")
    pdf.drawString(margin, 46, "Seal: packs/qwen3-30b-a3b-nhdf-v03-iq2m/evidence/functional_gate.json")
    pdf.drawRightString(width - margin, 57, "Measured 31 August 2026")
    pdf.drawRightString(width - margin, 46, "Page 1 of 1")

    pdf.showPage()
    pdf.save()
    return OUTPUT


if __name__ == "__main__":
    path = build()
    print(path)
