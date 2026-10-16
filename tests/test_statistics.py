"""Unit tests for khainglens.statistics covering metrics, distributions, rates, and empty datasets."""

import pytest

from khainglens.models import (
    AnalysisScope,
    Anomaly,
    AnomalyType,
    DuplicateGroup,
    InputFormat,
    LoadResult,
    MalformedRecord,
    ParsedRecord,
    Severity,
)
from khainglens.statistics import compute_dataset_statistics


def test_compute_dataset_statistics_comprehensive() -> None:
    """Verify correct aggregation of record counts, missing/empty rates, duplicates, and lengths."""
    valid_recs = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="မင်္ဂလာပါ မြန်မာစာ"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="မင်္ဂလာပါ မြန်မာစာ"),  # duplicate
        ParsedRecord(record_index=2, source_line_number=3, raw_text=""),                   # empty
        ParsedRecord(record_index=3, source_line_number=4, raw_text=None, is_missing_value=True),  # missing
        ParsedRecord(record_index=4, source_line_number=5, raw_text="   "),                # whitespace only
        ParsedRecord(record_index=5, source_line_number=6, raw_text="Short"),
    ]
    malformed_recs = [
        MalformedRecord(
            record_index=6,
            source_line_number=7,
            raw_excerpt="broken",
            reason="JSON error",
            error_detail="syntax",
        )
    ]

    load_res = LoadResult(
        filename="test.jsonl",
        file_format=InputFormat.JSONL,
        encoding_used="utf-8",
        has_bom=False,
        analysis_scope=AnalysisScope.COMPLETE,
        truncated_at_limit=False,
        max_records_limit_applied=100_000,
        analyzed_raw_records=7,
        valid_records=valid_recs,
        malformed_records=malformed_recs,
    )

    dup_group = DuplicateGroup(
        group_id=1,
        representative_excerpt="မင်္ဂလာပါ မြန်မာစာ",
        record_indices=[0, 1],
        occurrence_count=2,
        surplus_count=1,
    )
    iqr_bounds = {
        "q1": 5.0,
        "q3": 18.0,
        "iqr": 13.0,
        "lower_fence": -14.5,
        "upper_fence": 37.5,
    }

    stats = compute_dataset_statistics(
        load_res,
        anomalies=[],
        duplicate_groups=[dup_group],
        iqr_bounds=iqr_bounds,
    )

    assert stats.analyzed_raw_records == 7
    assert stats.successfully_parsed_records == 6
    assert stats.malformed_record_count == 1
    assert stats.missing_value_record_count == 1
    assert stats.empty_string_record_count == 1
    assert stats.whitespace_only_record_count == 1
    assert stats.total_empty_or_missing_records == 3
    assert stats.non_empty_analyzable_records == 3
    assert stats.unique_non_empty_texts == 2
    assert stats.duplicate_group_count == 1
    assert stats.total_records_in_duplicate_groups == 2
    assert stats.surplus_duplicate_occurrences == 1
    assert stats.malformed_record_rate == pytest.approx(1 / 7)
    assert stats.empty_or_missing_rate == pytest.approx(3 / 6)
    assert stats.duplicate_involvement_rate == pytest.approx(2 / 3)
    assert stats.surplus_duplicate_rate == pytest.approx(1 / 3)


def test_length_statistics_math_exactness() -> None:
    """Verify min, max, mean, and median length calculations on known code-point lengths."""
    # Code point lengths: 4, 10, 16, 30 -> min=4, max=30, mean=15.0, median=13.0
    valid_recs = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="A" * 4),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="B" * 10),
        ParsedRecord(record_index=2, source_line_number=3, raw_text="C" * 16),
        ParsedRecord(record_index=3, source_line_number=4, raw_text="D" * 30),
    ]
    load_res = LoadResult(
        filename="lengths.txt",
        file_format=InputFormat.TXT,
        encoding_used="utf-8",
        has_bom=False,
        analyzed_raw_records=4,
        valid_records=valid_recs,
        malformed_records=[],
    )

    stats = compute_dataset_statistics(
        load_res, anomalies=[], duplicate_groups=[], iqr_bounds={}
    )

    assert stats.min_length == 4
    assert stats.max_length == 30
    assert stats.mean_length == pytest.approx(15.0)
    assert stats.median_length == pytest.approx(13.0)
    assert sum(b.count for b in stats.length_histogram) == 4


def test_distinguish_anomaly_findings_occurrences_and_affected_records() -> None:
    """CRITICAL: Verify 3 anomalies on 1 record yield total_anomaly_findings=3 but records_with_any_anomaly=1."""
    valid_recs = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="  မြန်\u200bမာ\u200bစာ  "),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="Clean record"),
    ]
    load_res = LoadResult(
        filename="multi_anom.txt",
        file_format=InputFormat.TXT,
        encoding_used="utf-8",
        has_bom=False,
        analyzed_raw_records=2,
        valid_records=valid_recs,
        malformed_records=[],
    )

    anomalies = [
        Anomaly(
            anomaly_id="anom_000000_leading_whitespace",
            record_index=0,
            issue_type=AnomalyType.LEADING_WHITESPACE,
            severity=Severity.INFO,
            title="Leading Whitespace",
            explanation="...",
            original_excerpt="...",
            evidence="...",
            suggested_action="...",
            occurrence_count=2,
        ),
        Anomaly(
            anomaly_id="anom_000000_trailing_whitespace",
            record_index=0,
            issue_type=AnomalyType.TRAILING_WHITESPACE,
            severity=Severity.INFO,
            title="Trailing Whitespace",
            explanation="...",
            original_excerpt="...",
            evidence="...",
            suggested_action="...",
            occurrence_count=2,
        ),
        Anomaly(
            anomaly_id="anom_000000_zero_width_character",
            record_index=0,
            issue_type=AnomalyType.ZERO_WIDTH_CHARACTER,
            severity=Severity.WARNING,
            title="Zero-Width Characters",
            explanation="...",
            original_excerpt="...",
            evidence="...",
            suggested_action="...",
            occurrence_count=3,
        ),
    ]

    stats = compute_dataset_statistics(
        load_res, anomalies=anomalies, duplicate_groups=[], iqr_bounds={}
    )

    # 3 distinct Anomaly findings on Record #0, totaling 2 + 2 + 3 = 7 character occurrences
    assert stats.total_anomaly_findings == 3
    assert stats.total_in_record_anomaly_occurrences == 7
    # Only 1 unique record (Record #0) is affected!
    assert stats.records_with_any_anomaly == 1
    assert stats.affected_record_rate == pytest.approx(1 / 2)

    # Severity breakdown: 2 INFO findings on 1 unique record; 1 WARNING finding on 1 unique record
    assert stats.anomaly_counts_by_severity["info"] == 2
    assert stats.affected_records_by_severity["info"] == 1
    assert stats.anomaly_counts_by_severity["warning"] == 1
    assert stats.affected_records_by_severity["warning"] == 1


def test_empty_dataset_statistics_zero_safe() -> None:
    """Verify empty datasets (0 records) compute stats safely without ZeroDivisionError."""
    empty_load = LoadResult(
        filename="empty.txt",
        file_format=InputFormat.TXT,
        encoding_used="utf-8",
        has_bom=False,
        analysis_scope=AnalysisScope.COMPLETE,
        truncated_at_limit=False,
        max_records_limit_applied=100_000,
        analyzed_raw_records=0,
        valid_records=[],
        malformed_records=[],
    )
    stats = compute_dataset_statistics(
        empty_load, anomalies=[], duplicate_groups=[], iqr_bounds={}
    )
    assert stats.analyzed_raw_records == 0
    assert stats.min_length is None
    assert stats.max_length is None
    assert stats.mean_length is None
    assert stats.median_length is None
    assert stats.malformed_record_rate is None
    assert stats.empty_or_missing_rate is None
    assert stats.duplicate_involvement_rate is None
    assert stats.surplus_duplicate_rate is None
    assert stats.affected_record_rate is None


def test_truncated_subset_scope_propagated_to_stats() -> None:
    """Verify truncated_at_limit and TRUNCATED_SUBSET scope propagate into DatasetStats."""
    trunc_load = LoadResult(
        filename="partial.csv",
        file_format=InputFormat.CSV,
        encoding_used="utf-8",
        has_bom=False,
        analysis_scope=AnalysisScope.TRUNCATED_SUBSET,
        truncated_at_limit=True,
        max_records_limit_applied=50,
        analyzed_raw_records=50,
        valid_records=[],
        malformed_records=[],
    )
    stats = compute_dataset_statistics(
        trunc_load, anomalies=[], duplicate_groups=[], iqr_bounds={}
    )
    assert stats.truncated_at_limit is True
    assert stats.analysis_scope == AnalysisScope.TRUNCATED_SUBSET
    assert stats.max_records_limit_applied == 50