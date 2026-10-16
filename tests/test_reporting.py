"""Unit tests for khainglens.reporting covering JSON and CSV exports, Unicode, and truncation."""

import csv
import io
import json

from khainglens.analyzer import inspect_dataset_bytes
from khainglens.models import AnalysisConfig
from khainglens.reporting import (
    CSV_REPORT_COLUMNS,
    build_export_filename,
    generate_csv_report,
    generate_json_report,
)


def _create_sample_analysis_result(max_records_limit: int = 100_000):
    """Create a representative AnalysisResult with Burmese, quotes, commas, and anomalies."""
    csv_content = (
        "id,text\n"
        '1,"မင်္ဂလာပါ, ""မြန်မာစာ"" အချက်အလက်"\n'
        '2,"မင်္ဂလာပါ, ""မြန်မာစာ"" အချက်အလက်"\n'
        '3,"  Leading and trailing\u200b\u200b with comma, and ""quotes""  "\n'
        "4,\n"
        "5,Malformed,Extra,Column\n"
    ).encode("utf-8")

    cfg = AnalysisConfig(max_records_limit=max_records_limit, excerpt_max_chars=50)
    return inspect_dataset_bytes(csv_content, filename="burmese_sample.csv", config=cfg)


def test_json_report_validity_and_burmese_preservation() -> None:
    """Verify JSON export is valid JSON, preserves unescaped Burmese text, and matches stats."""
    result = _create_sample_analysis_result()
    json_str = generate_json_report(result, include_full_text=False)

    # Direct Burmese characters must appear in the JSON string (not \u1019 escapes)
    assert "မင်္ဂလာပါ" in json_str

    parsed = json.loads(json_str)
    assert parsed["report_schema_version"] == "0.1.0"
    assert parsed["tool"]["name"] == "KhaingLens"
    assert parsed["input_metadata"]["filename"] == "burmese_sample.csv"
    assert parsed["input_metadata"]["analysis_scope"] == "complete_dataset"
    assert parsed["input_metadata"]["truncated_at_limit"] is False
    assert "COMPLETE DATASET ANALYSIS" in parsed["input_metadata"]["scope_notice"]

    # Statistics in JSON must match AnalysisResult.stats
    assert parsed["summary_statistics"]["analyzed_raw_records"] == 5
    assert parsed["summary_statistics"]["malformed_record_count"] == 1
    assert parsed["summary_statistics"]["duplicate_group_count"] == 1
    assert len(parsed["anomalies"]) == result.stats.total_anomaly_findings

    # By default, full_original_text is not included
    assert "full_original_text" not in parsed["anomalies"][0]


def test_json_report_include_full_text_toggle() -> None:
    """Verify include_full_text=True attaches full_original_text to exported anomalies."""
    long_burmese = "မင်္ဂလာပါ " * 30  # > 50 chars excerpt limit
    txt_bytes = f"  {long_burmese}  \n".encode("utf-8")
    cfg = AnalysisConfig(excerpt_max_chars=50)
    result = inspect_dataset_bytes(txt_bytes, filename="long.txt", config=cfg)

    json_default = json.loads(generate_json_report(result, include_full_text=False))
    anom_default = json_default["anomalies"][0]
    assert len(anom_default["original_excerpt"]) <= 50
    assert "full_original_text" not in anom_default

    json_full = json.loads(generate_json_report(result, include_full_text=True))
    anom_full = json_full["anomalies"][0]
    assert anom_full["full_original_text"] == f"  {long_burmese}  "


def test_csv_report_escaping_commas_quotes_and_burmese() -> None:
    """Verify CSV export preserves Burmese Unicode and escapes commas and quotes properly."""
    result = _create_sample_analysis_result()

    # 1. With UTF-8 BOM (default for Excel compatibility)
    csv_bytes_bom = generate_csv_report(result, include_utf8_bom=True, include_full_text=False)
    assert csv_bytes_bom.startswith(b"\xef\xbb\xbf")

    decoded = csv_bytes_bom.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(decoded))
    assert reader.fieldnames == CSV_REPORT_COLUMNS

    rows = list(reader)
    assert len(rows) == result.stats.total_anomaly_findings
    assert all(r["analysis_scope"] == "complete_dataset" for r in rows)
    assert all(r["truncated_at_limit"] == "False" for r in rows)

    # Verify Burmese text and embedded quotes/commas round-trip intact
    excerpts = [r["original_excerpt"] for r in rows]
    assert any('မင်္ဂလာပါ, "မြန်မာစာ"' in ex for ex in excerpts)


def test_csv_report_without_bom_and_with_full_text() -> None:
    """Verify CSV export supports include_utf8_bom=False and include_full_text=True."""
    result = _create_sample_analysis_result()
    csv_bytes = generate_csv_report(result, include_utf8_bom=False, include_full_text=True)

    assert not csv_bytes.startswith(b"\xef\xbb\xbf")
    decoded = csv_bytes.decode("utf-8")
    reader = csv.DictReader(io.StringIO(decoded))
    assert "full_original_text" in (reader.fieldnames or [])


def test_truncated_dataset_scope_in_json_and_csv_reports() -> None:
    """Verify truncated datasets explicitly mark truncated_subset in both JSON and CSV exports."""
    result = _create_sample_analysis_result(max_records_limit=2)

    assert result.load_result.truncated_at_limit is True

    # Check JSON report
    parsed_json = json.loads(generate_json_report(result))
    assert parsed_json["input_metadata"]["analysis_scope"] == "truncated_subset"
    assert parsed_json["input_metadata"]["truncated_at_limit"] is True
    assert "PARTIAL DATASET ANALYSIS" in parsed_json["input_metadata"]["scope_notice"]
    assert all(a["analysis_scope"] == "truncated_subset" for a in parsed_json["anomalies"])

    # Check CSV report
    csv_bytes = generate_csv_report(result, include_utf8_bom=False)
    rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8"))))
    assert len(rows) > 0
    assert all(r["analysis_scope"] == "truncated_subset" for r in rows)
    assert all(r["truncated_at_limit"] == "True" for r in rows)


def test_empty_dataset_reports_valid_json_and_header_only_csv() -> None:
    """Verify a dataset with 0 anomalies produces valid JSON and a valid header-only CSV."""
    clean_bytes = "Clean single sentence without anomalies.\n".encode("utf-8")
    result = inspect_dataset_bytes(clean_bytes, filename="clean.txt")

    assert result.stats.total_anomaly_findings == 0

    parsed_json = json.loads(generate_json_report(result))
    assert parsed_json["anomalies"] == []

    csv_bytes = generate_csv_report(result, include_utf8_bom=False)
    lines = csv_bytes.decode("utf-8").splitlines()
    assert len(lines) == 1  # Header row only
    assert lines[0].split(",") == CSV_REPORT_COLUMNS


def test_build_export_filename_sanitization() -> None:
    """Verify build_export_filename strips unsafe characters and appends timestamp and extension."""
    fname = build_export_filename(
        "My Burmese Corpus (v1).jsonl",
        extension=".json",
        timestamp_utc="2026-10-16T04:30:00Z",
    )
    assert fname == "khainglens_report_My_Burmese_Corpus_v1_20261016T043000Z.json"