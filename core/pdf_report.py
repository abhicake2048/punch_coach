"""Generate a polished, downloadable CornerCoach coaching report PDF."""

from __future__ import annotations

import html
import io
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .coaching_report import evidence_for_point


NAVY = colors.HexColor("#15253F")
BLUE = colors.HexColor("#2563EB")
PALE_BLUE = colors.HexColor("#EAF2FF")
PALE_GREEN = colors.HexColor("#EAF8F1")
PALE_AMBER = colors.HexColor("#FFF6DF")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#64748B")
GRID = colors.HexColor("#D7DFEA")


def _ascii(value: Any) -> str:
    """Return ReportLab-safe text without unsupported font glyphs."""
    text = str(value if value is not None else "")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2192", "to").replace("\u2022", "-")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _paragraph(value: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(html.escape(_ascii(value)).replace("\n", "<br/>"), style)


def _number(value: Any, digits: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{numeric:.{digits}f}"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CornerCoachTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "CornerCoachSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=MUTED,
            spaceAfter=4 * mm,
        ),
        "heading": ParagraphStyle(
            "CornerCoachHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=NAVY,
            spaceBefore=5 * mm,
            spaceAfter=2.5 * mm,
        ),
        "body": ParagraphStyle(
            "CornerCoachBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=INK,
            spaceAfter=2 * mm,
        ),
        "point_title": ParagraphStyle(
            "CornerCoachPointTitle",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
        ),
        "evidence": ParagraphStyle(
            "CornerCoachEvidence",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED,
        ),
        "metric_value": ParagraphStyle(
            "CornerCoachMetricValue",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=BLUE,
            alignment=TA_CENTER,
        ),
        "metric_label": ParagraphStyle(
            "CornerCoachMetricLabel",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "table": ParagraphStyle(
            "CornerCoachTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "CornerCoachTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def _metric_card(label: str, value: str, styles: Mapping[str, ParagraphStyle]) -> Table:
    card = Table(
        [
            [_paragraph(value, styles["metric_value"])],
            [_paragraph(label, styles["metric_label"])],
        ],
        colWidths=[42 * mm],
        rowHeights=[10 * mm, 8 * mm],
    )
    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return card


def _data_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    widths: Sequence[float],
    styles: Mapping[str, ParagraphStyle],
) -> Table:
    formatted = [[_paragraph(header, styles["table_header"]) for header in headers]]
    formatted.extend(
        [[_paragraph(value, styles["table"]) for value in row] for row in rows]
    )
    table = Table(formatted, colWidths=list(widths), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.5, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.8 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8 * mm),
            ]
        )
    )
    return table


def _coaching_points(
    title: str,
    points: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    styles: Mapping[str, ParagraphStyle],
    background: colors.Color,
) -> list[Any]:
    output: list[Any] = [_paragraph(title, styles["heading"])]
    for index, point in enumerate(points, start=1):
        evidence = evidence_for_point(point, metrics)
        evidence_text = ", ".join(f"{key} = {value}" for key, value in evidence.items())
        contents: list[Any] = [
            _paragraph(f"{index}. {point.get('title', '')}", styles["point_title"]),
            _paragraph(point.get("comment", ""), styles["body"]),
        ]
        if evidence_text:
            contents.append(_paragraph(f"Measured evidence: {evidence_text}", styles["evidence"]))
        card = Table([[contents]], colWidths=[174 * mm])
        card.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), background),
                    ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                    ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                ]
            )
        )
        output.extend([KeepTogether(card), Spacer(1, 2 * mm)])
    return output


def _footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "CornerCoach - metric-grounded coaching report")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def create_coaching_pdf(
    metrics: Mapping[str, Any],
    coaching_report: Mapping[str, Any],
    *,
    source_filename: str = "Uploaded video",
    generated_at: datetime | None = None,
) -> bytes:
    """Create a self-contained PDF from measured metrics and a validated LLM report."""
    styles = _styles()
    timestamp = generated_at or datetime.now(UTC)
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="CornerCoach Coaching Report",
        author="CornerCoach",
        subject="Boxing session analysis",
    )

    session = dict(metrics.get("session", {}))
    punches = dict(metrics.get("punches", {}))
    guard = dict(metrics.get("guard", {}))
    fatigue = dict(metrics.get("fatigue", {}))
    story: list[Any] = [
        _paragraph("CornerCoach Coaching Report", styles["title"]),
        _paragraph(
            f"Source: {_ascii(source_filename)} | Generated: {timestamp.strftime('%Y-%m-%d %H:%M UTC')} | "
            f"Model: {_ascii(session.get('model', 'Not available'))} | "
            f"Sequence: {_ascii(session.get('sequence_length_frames', 'Not available'))} frames",
            styles["subtitle"],
        ),
    ]

    cards = [
        _metric_card("TOTAL PUNCHES", str(int(punches.get("total", 0) or 0)), styles),
        _metric_card("PUNCHES / MIN", _number(punches.get("per_minute")), styles),
        _metric_card("GUARD SCORE", f"{_number(guard.get('discipline_score_percent'))}%", styles),
        _metric_card("FATIGUE", "Detected" if fatigue.get("detected") else "Stable", styles),
    ]
    card_row = Table([cards], colWidths=[44 * mm] * 4, hAlign="LEFT")
    card_row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm)]))
    story.extend([card_row, _paragraph("Session overview", styles["heading"]), _paragraph(coaching_report.get("summary", ""), styles["body"])])

    hand_counts = dict(punches.get("by_hand", {}))
    type_counts = dict(punches.get("by_type", {}))
    combination_counts = dict(punches.get("by_hand_and_type", {}))
    punch_rows: list[list[Any]] = []
    for label, count in sorted(type_counts.items()):
        punch_rows.append([str(label).title(), count])
    if not punch_rows:
        punch_rows.append(["No registered punches", 0])
    story.extend(
        [
            _paragraph("Punch analysis", styles["heading"]),
            _paragraph(
                f"Left hand: {int(hand_counts.get('left', 0) or 0)} | "
                f"Right hand: {int(hand_counts.get('right', 0) or 0)} | "
                f"Session duration: {_number(session.get('duration_seconds'))} seconds",
                styles["body"],
            ),
            _data_table(["Punch type", "Count"], punch_rows, [135 * mm, 39 * mm], styles),
        ]
    )
    if combination_counts:
        combination_rows = [[label, count] for label, count in sorted(combination_counts.items())]
        story.extend([Spacer(1, 2 * mm), _data_table(["Hand and type", "Count"], combination_rows, [135 * mm, 39 * mm], styles)])

    guard_events = list(guard.get("events", []))
    story.extend(
        [
            _paragraph("Guard discipline", styles["heading"]),
            _paragraph(
                f"Eligible frames: {int(guard.get('eligible_frames', 0) or 0)} | "
                f"Safe frames: {int(guard.get('safe_frames', 0) or 0)} | "
                f"Left score: {_number(dict(guard.get('per_arm_scores_percent', {})).get('left'))}% | "
                f"Right score: {_number(dict(guard.get('per_arm_scores_percent', {})).get('right'))}%",
                styles["body"],
            ),
        ]
    )
    if guard_events:
        event_rows = [
            [
                str(event.get("hand", "")).title(),
                _number(event.get("start_time_s"), 2),
                _number(event.get("end_time_s"), 2),
                _number(event.get("duration_s"), 2),
                str(event.get("deepest_level", "")).removeprefix("below_").replace("_", " ").title(),
            ]
            for event in guard_events
        ]
        story.append(_data_table(["Hand", "Start (s)", "End (s)", "Duration", "Deepest level"], event_rows, [30 * mm, 27 * mm, 27 * mm, 30 * mm, 60 * mm], styles))
    else:
        story.append(_paragraph("No guard-drop episodes were recorded on eligible frames.", styles["body"]))

    story.extend(
        [
            _paragraph("Fatigue comparison", styles["heading"]),
            _data_table(
                ["Segment", "Punches", "Work rate (PPM)", "Mean peak speed (TL/s)"],
                [
                    ["First third", fatigue.get("first_third_punches", 0), _number(fatigue.get("first_work_rate_ppm")), _number(fatigue.get("first_average_speed_torso_lengths_per_second"), 2)],
                    ["Last third", fatigue.get("last_third_punches", 0), _number(fatigue.get("last_work_rate_ppm")), _number(fatigue.get("last_average_speed_torso_lengths_per_second"), 2)],
                ],
                [43 * mm, 30 * mm, 50 * mm, 51 * mm],
                styles,
            ),
        ]
    )

    story.append(PageBreak())
    story.extend(_coaching_points("Three strengths", list(coaching_report.get("strengths", [])), metrics, styles, PALE_GREEN))
    story.extend(_coaching_points("Three areas to improve", list(coaching_report.get("areas_to_improve", [])), metrics, styles, PALE_AMBER))
    story.append(
        KeepTogether(
            _coaching_points(
                "Two suggested drills",
                list(coaching_report.get("suggested_drills", [])),
                metrics,
                styles,
                PALE_BLUE,
            )
        )
    )

    limitations = list(coaching_report.get("data_limitations", []))
    if limitations:
        story.append(_paragraph("Data limitations", styles["heading"]))
        for limitation in limitations:
            story.append(_paragraph(f"- {limitation}", styles["body"]))
    story.append(
        _paragraph(
            "This report summarizes model-generated measurements and coaching suggestions. "
            "It is not a substitute for an in-person qualified coach, medical advice, or safety supervision.",
            styles["subtitle"],
        )
    )

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
