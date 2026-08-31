"""Generate the durable NHDF hybrid validation PDF from sealed local evidence."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "packs" / "qwen3-30b-a3b-nhdf-v03-iq2m"
MANIFEST_PATH = PACK / "NHDF_HYBRID_MANIFEST.json"
MANIFEST_DIGEST_PATH = PACK / "NHDF_HYBRID_MANIFEST.sha256"
EVIDENCE_PATH = PACK / "evidence" / "functional_gate.json"
SOURCE_RECORD_PATH = (
    ROOT
    / "models"
    / "Qwen3-30B-A3B-Instruct-2507-IQ2_M"
    / "CONTROL_SOURCE.json"
)
RUNTIME_SOURCE_PATH = ROOT / "tools" / "llama.cpp-b6014" / "SOURCE.json"
OUTPUT = ROOT / "output" / "pdf" / "NHDF_Hybrid_Qwen3_30B_RTX5070Ti_Validation_Report.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1769AA")
TEAL = colors.HexColor("#0F766E")
GREEN = colors.HexColor("#18794E")
AMBER = colors.HexColor("#B7791F")
RED = colors.HexColor("#B42318")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_GREEN = colors.HexColor("#E8F5EE")
PALE_AMBER = colors.HexColor("#FFF7E6")
PALE_RED = colors.HexColor("#FDECEC")
LINE = colors.HexColor("#D9E2EC")
WHITE = colors.white


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


manifest = load_json(MANIFEST_PATH)
evidence = load_json(EVIDENCE_PATH)
source_record = load_json(SOURCE_RECORD_PATH)
runtime_source = load_json(RUNTIME_SOURCE_PATH)

manifest_digest = MANIFEST_DIGEST_PATH.read_text(encoding="ascii").strip()
evidence_digest = manifest["validation"]["evidence"]["sha256"]
payload = manifest["payload"]
aggregate = evidence["aggregate"]
benchmark = evidence["benchmark"]

bf16_bytes = int(manifest["model"]["source_bf16_tensor_bytes"])
payload_bytes = int(payload["bytes"])
bytes_saved = bf16_bytes - payload_bytes
ratio = bf16_bytes / payload_bytes
reduction_percent = (1.0 - payload_bytes / bf16_bytes) * 100.0
bf16_gib = bf16_bytes / (1024**3)
payload_gib = payload_bytes / (1024**3)
gpu_gib = int(aggregate["target_vram_mib"]) / 1024.0
peak_gib = int(aggregate["peak_gpu_memory_mib"]) / 1024.0
theoretical_4bit_gib = int(manifest["model"]["parameters"]) * 0.5 / (1024**3)


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=29,
        textColor=NAVY,
        alignment=TA_LEFT,
        spaceAfter=5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=MUTED,
        spaceAfter=6 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="H1x",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceBefore=2 * mm,
        spaceAfter=4 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="H2x",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=TEAL,
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Bodyx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=INK,
        spaceAfter=2.5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Smallx",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10,
        textColor=MUTED,
        spaceAfter=1.5 * mm,
    )
)
styles.add(
    ParagraphStyle(
        name="Calloutx",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=INK,
    )
)
styles.add(
    ParagraphStyle(
        name="CodeX",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=6.5,
        leading=8.5,
        textColor=INK,
    )
)


def para(text: str, style: str = "Bodyx") -> Paragraph:
    return Paragraph(text, styles[style])


def esc(value: object) -> str:
    return html.escape(str(value))


def section(title: str) -> Paragraph:
    return para(title, "H1x")


def subsection(title: str) -> Paragraph:
    return para(title, "H2x")


def callout(title: str, text: str, *, fill=PALE_BLUE, accent=BLUE) -> Table:
    body = para(f"<b>{esc(title)}</b><br/>{text}", "Bodyx")
    table = Table([[body]], colWidths=[166 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.5, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def data_table(rows: list[list[object]], widths: list[float], *, header: bool = True) -> Table:
    rendered: list[list[object]] = []
    for row_index, row in enumerate(rows):
        rendered.append(
            [
                value
                if hasattr(value, "wrap")
                else para(
                    esc(value),
                    "Smallx" if not (header and row_index == 0) else "Calloutx",
                )
                for value in row
            ]
        )
    table = Table(rendered, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
            ]
        )
    for row_index in range(1 if header else 0, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F7F9FB")))
    table.setStyle(TableStyle(commands))
    return table


def capacity_chart() -> Drawing:
    drawing = Drawing(470, 190)
    drawing.add(String(0, 174, "Storage and device capacity (GiB)", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    items = [
        ("BF16 tensor weights", bf16_gib, RED),
        ("Theoretical raw 4-bit", theoretical_4bit_gib, AMBER),
        ("GPU reported capacity", gpu_gib, NAVY),
        ("Measured 8K peak", peak_gib, TEAL),
        ("IQ2_M payload file", payload_gib, GREEN),
    ]
    x0 = 132
    usable = 310
    max_value = 60.0
    for index, (label, value, color) in enumerate(items):
        y = 143 - index * 28
        drawing.add(String(0, y + 3, label, fontName="Helvetica", fontSize=8, fillColor=INK))
        drawing.add(Rect(x0, y, usable, 12, fillColor=colors.HexColor("#EDF2F7"), strokeColor=None))
        drawing.add(Rect(x0, y, usable * value / max_value, 12, fillColor=color, strokeColor=None))
        drawing.add(String(x0 + usable + 7, y + 2, f"{value:.2f}", fontName="Helvetica-Bold", fontSize=8, fillColor=color))
    drawing.add(Line(x0, 18, x0 + usable, 18, strokeColor=LINE, strokeWidth=0.5))
    for tick in range(0, 61, 10):
        x = x0 + usable * tick / max_value
        drawing.add(Line(x, 16, x, 21, strokeColor=MUTED, strokeWidth=0.5))
        drawing.add(String(x - 4, 5, str(tick), fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    return drawing


def pipeline_diagram() -> Drawing:
    drawing = Drawing(470, 112)
    labels = [
        ("Pinned Qwen", "30.532B BF16", RED),
        ("External codec", "GGUF / IQ2_M", AMBER),
        ("NHDF substrate", "sealed + gated", TEAL),
        ("Pinned runtime", "llama.cpp b6014", BLUE),
        ("Target GPU", "RTX 5070 Ti", GREEN),
    ]
    box_w = 82
    gap = 14
    y = 35
    for index, (top, bottom, color) in enumerate(labels):
        x = index * (box_w + gap)
        drawing.add(Rect(x, y, box_w, 47, rx=5, ry=5, fillColor=colors.Color(color.red, color.green, color.blue, alpha=0.11), strokeColor=color, strokeWidth=1.2))
        drawing.add(String(x + box_w / 2, y + 29, top, fontName="Helvetica-Bold", fontSize=7.2, fillColor=color, textAnchor="middle"))
        drawing.add(String(x + box_w / 2, y + 15, bottom, fontName="Helvetica", fontSize=6.7, fillColor=INK, textAnchor="middle"))
        if index < len(labels) - 1:
            x1 = x + box_w + 2
            x2 = x + box_w + gap - 2
            drawing.add(Line(x1, y + 23, x2, y + 23, strokeColor=MUTED, strokeWidth=1))
            drawing.add(Line(x2 - 4, y + 26, x2, y + 23, strokeColor=MUTED, strokeWidth=1))
            drawing.add(Line(x2 - 4, y + 20, x2, y + 23, strokeColor=MUTED, strokeWidth=1))
    drawing.add(String(0, 95, "Responsibility boundary", fontName="Helvetica-Bold", fontSize=10, fillColor=NAVY))
    drawing.add(String(0, 17, "Compression belongs to Bartowski/ggml IQ2_M. NHDF supplies policy, binding, evidence, and fail-closed launch.", fontName="Helvetica", fontSize=7.3, fillColor=MUTED))
    return drawing


def code_block(text: str) -> Table:
    block = Preformatted(text, styles["CodeX"])
    table = Table([[block]], colWidths=[166 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6F9")),
                ("BOX", (0, 0), (-1, -1), 0.4, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def page_decor(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(22 * mm, height - 14 * mm, width - 22 * mm, height - 14 * mm)
    canvas.setFont("Helvetica", 6.7)
    canvas.setFillColor(MUTED)
    canvas.drawString(22 * mm, height - 10.5 * mm, "UGTOMS / NHDF hybrid validation record")
    canvas.drawRightString(width - 22 * mm, height - 10.5 * mm, "31 August 2026")
    canvas.line(22 * mm, 13 * mm, width - 22 * mm, 13 * mm)
    canvas.drawString(22 * mm, 8.5 * mm, "Evidence-scoped: functional smoke, full offload, allocated-8K residency, and throughput")
    canvas.drawRightString(width - 22 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_story() -> list[object]:
    story: list[object] = []
    story.extend(
        [
            Spacer(1, 19 * mm),
            para("NHDF Hybrid Validation Record", "ReportTitle"),
            para(
                "Complete Qwen3-30B-A3B inference on a 12 GB RTX 5070 Ti Laptop GPU",
                "ReportSubtitle",
            ),
            callout(
                "Final disposition: VALIDATED",
                "The complete 30,532,122,624-parameter model produced correct, coherent outputs through the normal fail-closed NHDF launcher. The externally authored IQ2_M payload peaked at 10,487 MiB on the 12,227 MiB GPU and measured 102.37 generated tokens/s in the final 64-token llama-bench run.",
                fill=PALE_GREEN,
                accent=GREEN,
            ),
            Spacer(1, 7 * mm),
            data_table(
                [
                    ["Document field", "Value"],
                    ["Artifact", "qwen3-30b-a3b-nhdf-v03-iq2m"],
                    ["Artifact format", manifest["format"]],
                    ["Validation scope", "Functional smoke, 49/49 GPU offload, allocated-8K residency, resource margin, and 64-token throughput"],
                    ["Created UTC", manifest["created_at_utc"]],
                    ["Gate UTC", evidence["generated_at_utc"]],
                    ["Target", manifest["resource_contract"]["target_gpu"]],
                    ["Manifest SHA-256", manifest_digest],
                    ["Evidence SHA-256", evidence_digest],
                ],
                [39 * mm, 127 * mm],
            ),
            Spacer(1, 8 * mm),
            para(
                "Authoritative evidence lives in the JSON manifest and functional gate record. This PDF is a human-readable technical record generated from those files.",
                "Smallx",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            section("1. Outcome and literal accessibility benefit"),
            para(
                "The target was not merely to open a model file. Success required the complete model to produce useful language, remain inside the laptop GPU resource envelope, and meet a declared generation-speed floor through an NHDF-controlled launch path. That target was achieved using an external mixed-bit codec inside an NHDF v0.3-informed substrate artifact.",
            ),
            capacity_chart(),
            data_table(
                [
                    ["Measurement", "Final value", "Interpretation"],
                    ["Model parameters", f"{manifest['model']['parameters']:,}", "Complete Qwen3-30B-A3B state; about 3.3B parameters are active per token."],
                    ["BF16 tensor bytes", f"{bf16_bytes:,} ({bf16_gib:.2f} GiB)", "Weights alone are 4.76x the reported GPU capacity; practical BF16 inference requires a 64 GiB-class device."],
                    ["IQ2_M payload file", f"{payload_bytes:,} ({payload_gib:.2f} GiB)", f"{ratio:.4f}x smaller; {reduction_percent:.2f}% reduction."],
                    ["Bytes removed", f"{bytes_saved:,}", "47.68 GiB fewer serialized weight bytes than BF16 tensor data."],
                    ["Measured allocated-8K peak", f"{aggregate['peak_gpu_memory_mib']:,} MiB", f"Leaves {aggregate['headroom_mib']:,} MiB of the reported {aggregate['target_vram_mib']:,} MiB."],
                    ["Generation microbenchmark", f"{benchmark['generation']['average_tokens_per_second']:.2f} +/- {benchmark['generation']['standard_deviation_tokens_per_second']:.2f} tok/s", "Three 64-token llama-bench samples; exceeds the fixed 80 tok/s floor."],
                ],
                [43 * mm, 48 * mm, 75 * mm],
            ),
            Spacer(1, 4 * mm),
            callout(
                "Accessibility in plain terms",
                "A workload whose BF16 weights alone need 56.87 GiB now runs on this laptop's 11.94 GiB reported device capacity. Even an ideal raw 4-bit representation would be 14.22 GiB before metadata and runtime buffers, so a conventional 4-bit floor would still miss the target.",
                fill=PALE_BLUE,
                accent=BLUE,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            section("2. What NHDF did - and did not do"),
            pipeline_diagram(),
            para(
                "The weight encoding is GGUF/IQ2_M from Bartowski and ggml. It is executed by llama.cpp b6014. NHDF does not claim authorship of that codec. In this result, NHDF is the controlling substrate: it binds exact artifacts and runtime files, records typed validation state, enforces GPU/context policy, stores sealed evidence, verifies its event chain, and refuses ordinary execution until promotion evidence passes.",
            ),
            data_table(
                [
                    ["Responsibility", "Owner", "Concrete implementation"],
                    ["Low-bit weight encoding", "Bartowski / ggml", "GGUF IQ2_M mixed-bit payload, explicitly attributed in the manifest."],
                    ["Tensor execution", "llama.cpp", "Pinned b6014 CUDA build for compute capability 12.0."],
                    ["Artifact binding", "NHDF hybrid", "Path, byte count, SHA-256, source revision, runtime file hashes, and specification hash."],
                    ["Resource policy", "NHDF hybrid", "Exact GPU identity, context <= 8192, q8 K/V, full offload, 10,712 MiB free-memory preflight, and 512 MiB reserve."],
                    ["Promotion", "NHDF hybrid", "UNCALIBRATED until functional, residency, offload, resource, and throughput gates all pass."],
                    ["Deployment launch", "NHDF hybrid", "Fails closed for altered, missing, unvalidated, wrong-GPU, or under-resourced artifacts."],
                ],
                [40 * mm, 35 * mm, 91 * mm],
            ),
            Spacer(1, 4 * mm),
            callout(
                "Honest boundary",
                "This is a successful NHDF-controlled hybrid execution profile. It is not proof of an NHDF-native weight codec, broad benchmark accuracy, or quality after filling all 8K context positions.",
                fill=PALE_AMBER,
                accent=AMBER,
            ),
            PageBreak(),
        ]
    )

    functional_rows: list[list[object]] = [["Gate", "Required", "Observed", "Result"]]
    for result in evidence["functional_results"]:
        rule = result["acceptance_rule"]
        required = rule.get("value", "required terms")
        functional_rows.append(
            [result["id"], required, result["generated_text"], "PASS" if result["passed"] else "FAIL"]
        )
    story.extend(
        [
            section("3. Fresh validation procedure and outputs"),
            subsection("3.1 Promotion sequence"),
            para(
                "The final artifact was recreated in UNCALIBRATED state. A normal run was confirmed to refuse launch. The gate then performed a full payload hash check, five cold model launches, four 512-token functional prompts, one allocated-8192 cache run with a short exact-response prompt, and three prompt plus three generation benchmark samples. Only the complete pass wrote VALIDATED.",
            ),
            data_table(functional_rows, [31 * mm, 35 * mm, 82 * mm, 18 * mm]),
            Spacer(1, 3 * mm),
            subsection("3.2 Residency and buffer observations"),
            data_table(
                [
                    ["Resource", "Measured value"],
                    ["Physical GPU capacity", f"{aggregate['target_vram_mib']:,} MiB"],
                    ["Idle baseline in canonical run", f"{evidence['allocated_8k_residency_result']['baseline_gpu_memory_mib']:,} MiB"],
                    ["Peak system-wide device use", f"{aggregate['peak_gpu_memory_mib']:,} MiB"],
                    ["Measured headroom", f"{aggregate['headroom_mib']:,} MiB"],
                    ["CUDA model buffer", f"{evidence['allocated_8k_residency_result']['llama_metrics']['cuda_model_buffer_mib']:.2f} MiB"],
                    ["CUDA q8 K/V buffer", f"{evidence['allocated_8k_residency_result']['llama_metrics']['cuda_kv_buffer_mib']:.2f} MiB"],
                    ["CUDA compute buffer", f"{evidence['allocated_8k_residency_result']['llama_metrics']['cuda_compute_buffer_mib']:.2f} MiB"],
                    ["CPU-mapped model buffer", f"{evidence['allocated_8k_residency_result']['llama_metrics']['cpu_mapped_model_buffer_mib']:.2f} MiB"],
                    ["Reported offload", f"{evidence['allocated_8k_residency_result']['llama_metrics']['offloaded_layers'][0]}/{evidence['allocated_8k_residency_result']['llama_metrics']['offloaded_layers'][1]} layers"],
                ],
                [79 * mm, 87 * mm],
            ),
            Spacer(1, 3 * mm),
            para(
                "The 8K gate allocates an 8192-position q8 K/V cache and executes a short prompt. It proves allocation, residency, offload, and short execution at that capacity. It does not measure filled-8K retrieval or answer quality.",
                "Smallx",
            ),
            PageBreak(),
        ]
    )

    prompt_samples = benchmark["prompt"]["samples_tokens_per_second"]
    generation_samples = benchmark["generation"]["samples_tokens_per_second"]
    story.extend(
        [
            section("4. Throughput and operational behavior"),
            data_table(
                [
                    ["Workload", "Samples (tok/s)", "Mean", "Std. dev.", "Gate"],
                    ["64-token prompt", ", ".join(f"{value:.3f}" for value in prompt_samples), f"{benchmark['prompt']['average_tokens_per_second']:.6f}", f"{benchmark['prompt']['standard_deviation_tokens_per_second']:.6f}", "Recorded"],
                    ["64-token generation", ", ".join(f"{value:.4f}" for value in generation_samples), f"{benchmark['generation']['average_tokens_per_second']:.6f}", f"{benchmark['generation']['standard_deviation_tokens_per_second']:.6f}", ">= 80 PASS"],
                ],
                [34 * mm, 58 * mm, 29 * mm, 27 * mm, 18 * mm],
            ),
            Spacer(1, 4 * mm),
            callout(
                "Interpretation of 102.37 tok/s",
                "This is a llama-bench batch-one, 64-token generation microbenchmark on a mixture-of-experts model. All 30.532B parameters are stored, but about 3.3B are active per token. The result is therefore not dense-30B throughput and is not an 'optimal under every workload' claim.",
                fill=PALE_BLUE,
                accent=BLUE,
            ),
            Spacer(1, 4 * mm),
            subsection("4.1 Normal validated launch"),
            para(
                "After promotion and full byte-for-byte verification, the ordinary launcher was run without --allow-unvalidated. It returned:",
            ),
            code_block(
                "The validated NHDF hybrid enables efficient, high-performance neural network\n"
                "inference directly on this laptop's GPU."
            ),
            Spacer(1, 4 * mm),
            subsection("4.2 Fail-closed resource behavior observed during development"),
            para(
                "One intermediate retry detected a transient system-wide GPU allocation during the 8K launch: peak use reached 12,088 MiB and only 139 MiB remained. Functional prompts and throughput still passed, but promotion was rejected because the 512 MiB reserve failed. Idle use then returned to 287 MiB; the artifact was recreated and the final canonical gate passed at 10,487 MiB. This observation is not the canonical certificate, but it demonstrates why system-wide peak sampling and fail-closed resource disposition matter.",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            section("5. What failed before the successful hybrid"),
            subsection("5.1 Legacy NHDF-native scalar pack - functional failure"),
            para(
                "The original 9,152,386,624-byte native tensor pack loaded, passed 531 manifest/CRC/parity checks, fit the GPU, and executed CUDA generation. Its output collapsed to repeated newline tokens or the string 10000000. It is sealed as QUALITY_FAILED and the native loader refuses it by default. Integrity and fit were real; language quality failed.",
            ),
            data_table(
                [
                    ["Layer-0 expert", "Output NRMSE", "Cosine similarity"],
                    ["0", "0.4813", "0.8805"],
                    ["17", "0.4749", "0.8935"],
                    ["127", "0.4482", "0.9170"],
                ],
                [55 * mm, 55 * mm, 56 * mm],
            ),
            Spacer(1, 4 * mm),
            subsection("5.2 Replacement native codec - comparative failure, not collapse"),
            para(
                "A later routed-calibrated 3-bit GPTQ experiment achieved useful numerical behavior. On the complete router-weighted layer-0 gate over 656 disjoint holdout tokens and all 117 experts selected by real top-8 routes, its NRMSE was 0.155101 versus 0.150849 for simpler equal-storage RTN. The candidate was 2.819% worse. Absolute quality passed, but the predeclared comparative-advantage gate failed, so a native full pack was not justified.",
            ),
            callout(
                "Correct conclusion",
                "The native scalar format genuinely failed language generation. The replacement native candidate did not collapse, but it failed to beat a simpler equal-budget codec. Neither negative result invalidates the separately validated external-codec hybrid.",
                fill=PALE_RED,
                accent=RED,
            ),
            PageBreak(),
        ]
    )

    create_command = r'''nhdf-edge create-hybrid packs\qwen3-30b-a3b-nhdf-v03-iq2m `
  --model models\Qwen3-30B-A3B-Instruct-2507-IQ2_M\Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf `
  --runtime tools\llama.cpp-b6014\bin\llama-cli.exe `
  --benchmark-runtime tools\llama.cpp-b6014\bin\llama-bench.exe `
  --specification sources\NHDF_Formal_Specification_v0.3_General_Purpose_CCD_Tom_Klootwijk.pdf `
  --source-record models\Qwen3-30B-A3B-Instruct-2507-IQ2_M\CONTROL_SOURCE.json `
  --assurance-evidence metrics\local\gguf_backend_ops.json

nhdf-edge gate-hybrid packs\qwen3-30b-a3b-nhdf-v03-iq2m
nhdf-edge verify packs\qwen3-30b-a3b-nhdf-v03-iq2m
nhdf-edge run packs\qwen3-30b-a3b-nhdf-v03-iq2m --quick --text-only `
  --prompt "Reply with exactly the single word OK." --max-new-tokens 8'''

    story.extend(
        [
            section("6. Reproduce the validated path"),
            subsection("6.1 Restore large external inputs"),
            para(
                "The Git repository deliberately excludes the 9.87 GB GGUF, 61.07 GB BF16 source, build trees, temporary calibration data, and profiler captures. The exact validated llama.cpp binaries, their license, and their provenance record are included because the sealed runtime set remains below the repository publication budget. Restore only the IQ2_M model file at the path below and verify both byte count and SHA-256 before creating the hybrid.",
            ),
            data_table(
                [
                    ["Input", "Pinned value"],
                    ["Model repository", source_record["source_repository"]],
                    ["Model revision", source_record["source_revision"]],
                    ["Model artifact", source_record["artifact"]],
                    ["Model bytes", f"{source_record['bytes']:,}"],
                    ["Model SHA-256", source_record["sha256"]],
                    ["llama.cpp source", runtime_source["source_url"]],
                    ["llama.cpp source archive SHA-256", runtime_source["archive_sha256"]],
                    ["Build profile", f"Ninja; CUDA {runtime_source['build']['cuda_toolkit']}; SM {runtime_source['build']['cuda_architecture']}; {runtime_source['build']['compiler']}"],
                ],
                [49 * mm, 117 * mm],
            ),
            Spacer(1, 3 * mm),
            para(
                "Use scripts/download_verified_ranges.ps1 with the immutable_url, byte count, and hash in CONTROL_SOURCE.json when segmented downloading is useful.",
                "Smallx",
            ),
            subsection("6.2 Create, gate, verify, and run"),
            code_block(create_command),
            Spacer(1, 4 * mm),
            para(
                "A new artifact begins UNCALIBRATED. Do not use --allow-unvalidated for deployment. The final gate must be rerun on the target machine because VRAM headroom and throughput are hardware- and background-load-dependent.",
            ),
            PageBreak(),
        ]
    )

    runtime_rows = [["Sealed runtime file", "Bytes", "SHA-256"]]
    for record in manifest["runtime"]["files"]:
        runtime_rows.append([Path(record["path"]).name, f"{record['bytes']:,}", record["sha256"]])
    story.extend(
        [
            section("Appendix A. Identity and integrity records"),
            data_table(
                [
                    ["Record", "Value"],
                    ["Upstream Qwen revision", manifest["model"]["source_revision"]],
                    ["GGUF revision", source_record["source_revision"]],
                    ["GGUF SHA-256", payload["sha256"]],
                    ["NHDF v0.3 specification SHA-256", manifest["substrate"]["specification"]["sha256"]],
                    ["Manifest SHA-256", manifest_digest],
                    ["Functional evidence SHA-256", evidence_digest],
                    ["Event-chain records verified", str(len(manifest["events"]))],
                    ["Backend assurance", "8,132 / 8,132 CUDA MUL_MAT_ID cases passed; separately sealed assurance evidence"],
                ],
                [56 * mm, 110 * mm],
            ),
            Spacer(1, 4 * mm),
            data_table(runtime_rows, [43 * mm, 25 * mm, 98 * mm]),
            Spacer(1, 4 * mm),
            para(
                "These are SHA-256-sealed local records, not externally signed attestations. Recomputing both a file and its local digest is possible for a repository writer; use signed Git commits or an external transparency log if adversarial provenance becomes a requirement.",
                "Smallx",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            section("Appendix B. Evidence boundary and next work"),
            subsection("Established on this machine"),
            para(
                "- Complete-model loading and coherent generation through the NHDF launcher.<br/>"
                "- Exact payload, runtime, specification, source-record, and evidence hashes.<br/>"
                "- Four of four declared functional prompts passed.<br/>"
                "- Forty-nine of forty-nine layers offloaded in every canonical launch.<br/>"
                "- Allocated-8K q8 K/V residency at 10,487 MiB peak with 1,740 MiB headroom.<br/>"
                "- Final 64-token generation microbenchmark at 102.367894 +/- 3.590717 tok/s.<br/>"
                "- Fail-closed status, GPU identity, free-memory, context, reserve, and throughput policy.<br/>"
                "- Sixty-two Python tests passed after the final implementation changes."
            ),
            subsection("Not established"),
            para(
                "- Broad benchmark accuracy, perplexity, or KL agreement against BF16.<br/>"
                "- Retrieval or language quality after actually filling 8K context positions.<br/>"
                "- Sustained thermals, power, p10/p90 performance, or long-duration stability.<br/>"
                "- Dense-30B throughput equivalence.<br/>"
                "- An NHDF-native tensor-codec advantage.<br/>"
                "- A self-contained portable artifact; the validated zero-copy manifest is workspace-bound.<br/>"
                "- Performance portability to other GPUs or llama.cpp revisions."
            ),
            subsection("Recommended next gates"),
            data_table(
                [
                    ["Priority", "Next gate", "Why"],
                    ["1", "Broad quality suite against BF16 or a stronger reference", "Turns the narrow smoke certificate into an evidence-backed quality statement."],
                    ["2", "Filled-context tests at 2K, 4K, and 8K", "Separates allocated capacity from usable long-context quality and throughput."],
                    ["3", "Sustained 10-30 minute thermal and power runs", "Quantifies laptop throttling and stable tokens/s rather than cold microbenchmarks."],
                    ["4", "Signed release manifest or external transparency record", "Strengthens provenance beyond local digest files."],
                    ["5", "Native-codec calibration-only selector versus RTN", "Continue native research only where a candidate beats the simpler equal-budget baseline."],
                ],
                [18 * mm, 63 * mm, 85 * mm],
            ),
            Spacer(1, 5 * mm),
            callout(
                "Bottom line",
                "The practical objective was achieved: the complete 30.532B Qwen model is accessible and useful on the tested 12 GB laptop GPU through an NHDF-controlled, externally encoded hybrid. The exact success, failures, hashes, commands, measurements, and limitations are preserved here for the next iteration.",
                fill=PALE_GREEN,
                accent=GREEN,
            ),
        ]
    )
    return story


def main() -> None:
    required = [MANIFEST_PATH, MANIFEST_DIGEST_PATH, EVIDENCE_PATH, SOURCE_RECORD_PATH, RUNTIME_SOURCE_PATH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required evidence missing: {missing}")
    if sha256_file(MANIFEST_PATH) != manifest_digest:
        raise ValueError("manifest digest mismatch before PDF generation")
    if sha256_file(EVIDENCE_PATH) != evidence_digest:
        raise ValueError("evidence digest mismatch before PDF generation")
    if manifest["validation"]["status"] != "VALIDATED" or evidence.get("passed") is not True:
        raise ValueError("refusing to publish a success report for an unvalidated artifact")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title="NHDF Hybrid Qwen3-30B RTX 5070 Ti Validation Report",
        author="Tom Klootwijk / UGTOMS",
        subject="Measured functional and resource validation record",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="content")
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=page_decor))
    doc.build(build_story())
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "bytes": OUTPUT.stat().st_size,
                "sha256": sha256_file(OUTPUT),
                "status": "created",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
