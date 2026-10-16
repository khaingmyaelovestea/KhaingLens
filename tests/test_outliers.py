"""Unit tests for khainglens.outlier_detection covering IQR fences, small datasets, and edge cases."""

from khainglens.models import AnalysisConfig, AnomalyType, ParsedRecord, Severity
from khainglens.outlier_detection import detect_length_outliers


def test_iqr_length_outlier_detection() -> None:
    """Verify IQR outlier detection correctly flags unusually short and long records."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="A" * 10),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="A" * 11),
        ParsedRecord(record_index=2, source_line_number=3, raw_text="A" * 12),
        ParsedRecord(record_index=3, source_line_number=4, raw_text="A" * 12),
        ParsedRecord(record_index=4, source_line_number=5, raw_text="A" * 13),
        ParsedRecord(record_index=5, source_line_number=6, raw_text="A" * 14),
        # Extremely short outlier
        ParsedRecord(record_index=6, source_line_number=7, raw_text="Hi"),
        # Extremely long outlier
        ParsedRecord(record_index=7, source_line_number=8, raw_text="A" * 100),
    ]

    bounds, anomalies = detect_length_outliers(records)

    assert bounds["q1"] is not None
    assert bounds["q3"] is not None
    assert bounds["iqr"] is not None
    assert bounds["lower_fence"] is not None
    assert bounds["upper_fence"] is not None

    types = {a.issue_type for a in anomalies}
    assert AnomalyType.LENGTH_OUTLIER_SHORT in types
    assert AnomalyType.LENGTH_OUTLIER_LONG in types

    short_anom = next(a for a in anomalies if a.issue_type == AnomalyType.LENGTH_OUTLIER_SHORT)
    long_anom = next(a for a in anomalies if a.issue_type == AnomalyType.LENGTH_OUTLIER_LONG)

    assert short_anom.record_index == 6
    assert short_anom.severity == Severity.INFO
    assert long_anom.record_index == 7
    assert long_anom.severity == Severity.INFO


def test_small_dataset_skips_iqr_safely() -> None:
    """Verify datasets smaller than min_records_for_iqr (default 4) skip IQR without error."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="Short"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="Longer sentence"),
    ]
    bounds, anomalies = detect_length_outliers(records)
    assert bounds["q1"] is None
    assert anomalies == []


def test_uniform_length_dataset_iqr_zero_skips_outliers() -> None:
    """Verify datasets where all records have identical length (IQR == 0) do not flag false outliers."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="Exact length"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="Exact length"),
        ParsedRecord(record_index=2, source_line_number=3, raw_text="Exact length"),
        ParsedRecord(record_index=3, source_line_number=4, raw_text="Exact length"),
    ]
    bounds, anomalies = detect_length_outliers(records)
    assert bounds["iqr"] == 0.0
    assert anomalies == []