"""CSV and JSON inspection report generation for KhaingLens."""

import csv
import io
import json
from pathlib import Path
import re
from typing import Any

from khainglens import __app_name__, __version__
from khainglens.analyzer import build_record_lookup
from khainglens.models import (
    AnalysisResult,
    AnalysisScope,
    MalformedRecord,
    ParsedRecord,
)

CSV_REPORT_COLUMNS: list[str] = [
    "anomaly_id",
    "analysis_scope",
    "truncated_at_limit",
    "record_index",
    "source_line_number",
    "issue_type",
    "severity",
    "occurrence_count",
    "first_char_position",
    "all_char_positions",
    "title",
    "explanation",
    "evidence",
    "original_excerpt",
    "suggested_action",
    "metadata_json",
]


def _build_scope_notice(result: AnalysisResult) -> str:
    """Return an explicit human-readable explanation of the analysis scope."""
    lr = result.load_result
    if lr.truncated_at_limit or lr.analysis_scope == AnalysisScope.TRUNCATED_SUBSET:
        return (
            f"PARTIAL DATASET ANALYSIS (TRUNCATED AT LIMIT): Only the first "
            f"{lr.analyzed_raw_records:,} records (limit: {lr.max_records_limit_applied:,}) "
            "were analyzed. Statistics, duplicate groups, and anomaly counts apply strictly "
            "to this analyzed subset and must not be interpreted as complete-dataset results."
        )
    return (
        f"COMPLETE DATASET ANALYSIS: All {lr.analyzed_raw_records:,} records in "
        f"'{lr.filename}' were inspected."
    )


def _resolve_full_text(
    record_obj: ParsedRecord | MalformedRecord | None,
) -> str | None:
    """Extract raw text or malformed excerpt for optional full-text report export."""
    if record_obj is None:
        return None
    if isinstance(record_obj, ParsedRecord):
        return record_obj.raw_text
    return record_obj.raw_excerpt


def build_export_filename(
    dataset_filename: str,
    extension: str,
    timestamp_utc: str | None = None,
) -> str:
    """Generate a clean, deterministic filename for downloaded CSV or JSON reports."""
    stem = Path(dataset_filename).stem if dataset_filename else "dataset"
    safe_stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem).strip("_") or "dataset"
    ext = extension.lstrip(".").lower()
    if timestamp_utc:
        compact_ts = (
            timestamp_utc.replace("-", "").replace(":", "").replace(" ", "_")
        )
        return f"khainglens_report_{safe_stem}_{compact_ts}.{ext}"
    return f"khainglens_report_{safe_stem}.{ext}"


def generate_json_report(
    result: AnalysisResult,
    include_full_text: bool = False,
) -> str:
    """Serialize an AnalysisResult into a structured UTF-8 JSON report string.

    Preserves Burmese and international Unicode characters directly without ASCII escaping.
    """
    lr = result.load_result
    lookup = build_record_lookup(lr) if include_full_text else {}

    serialized_anomalies: list[dict[str, Any]] = []
    for anom in result.anomalies:
        anom_dict = anom.model_dump(mode="json")
        anom_dict["analysis_scope"] = lr.analysis_scope.value
        anom_dict["truncated_at_limit"] = lr.truncated_at_limit
        if include_full_text:
            anom_dict["full_original_text"] = _resolve_full_text(
                lookup.get(anom.record_index)
            )
        serialized_anomalies.append(anom_dict)

    report_payload: dict[str, Any] = {
        "report_schema_version": __version__,
        "generated_at_utc": result.analysis_timestamp_utc,
        "tool": {
            "name": __app_name__,
            "version": __version__,
        },
        "input_metadata": {
            "filename": lr.filename,
            "file_format": lr.file_format.value,
            "encoding_used": lr.encoding_used,
            "has_bom": lr.has_bom,
            "csv_has_header": lr.csv_has_header,
            "selected_field": lr.selected_field,
            "detected_delimiter": lr.detected_delimiter,
            "analysis_scope": lr.analysis_scope.value,
            "truncated_at_limit": lr.truncated_at_limit,
            "max_records_limit_applied": lr.max_records_limit_applied,
            "scope_notice": _build_scope_notice(result),
            "text_inclusion_policy": (
                "full_original_text_and_bounded_excerpt"
                if include_full_text
                else f"bounded_excerpt_only (max {result.config.excerpt_max_chars} chars)"
            ),
            "load_warnings": lr.load_warnings,
        },
        "analysis_config": result.config.model_dump(mode="json"),
        "summary_statistics": result.stats.model_dump(mode="json"),
        "duplicate_groups": [
            g.model_dump(mode="json") for g in result.duplicate_groups
        ],
        "anomalies": serialized_anomalies,
    }

    return json.dumps(report_payload, ensure_ascii=False, indent=2)


def generate_csv_report(
    result: AnalysisResult,
    include_utf8_bom: bool = True,
    include_full_text: bool = False,
) -> bytes:
    """Serialize all consolidated anomalies from AnalysisResult into UTF-8 CSV bytes.

    By default includes a UTF-8 BOM (utf-8-sig) so spreadsheet programs such as
    Microsoft Excel open Burmese Unicode text without character corruption.
    """
    lr = result.load_result
    lookup = build_record_lookup(lr) if include_full_text else {}

    columns = list(CSV_REPORT_COLUMNS)
    if include_full_text:
        columns.append("full_original_text")

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    writer.writerow(columns)

    for anom in result.anomalies:
        row: list[Any] = [
            anom.anomaly_id,
            lr.analysis_scope.value,
            lr.truncated_at_limit,
            anom.record_index,
            anom.source_line_number if anom.source_line_number is not None else "",
            anom.issue_type.value,
            anom.severity.value,
            anom.occurrence_count,
            anom.char_position if anom.char_position is not None else "",
            json.dumps(anom.char_positions, ensure_ascii=False),
            anom.title,
            anom.explanation,
            anom.evidence,
            anom.original_excerpt,
            anom.suggested_action,
            json.dumps(anom.metadata, ensure_ascii=False),
        ]
        if include_full_text:
            full_txt = _resolve_full_text(lookup.get(anom.record_index))
            row.append(full_txt if full_txt is not None else "<MISSING / NULL>")
        writer.writerow(row)

    csv_text = buffer.getvalue()
    encoding = "utf-8-sig" if include_utf8_bom else "utf-8"
    return csv_text.encode(encoding)