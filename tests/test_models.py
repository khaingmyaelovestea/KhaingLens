"""Unit tests for khainglens.models schemas, validation, and immutability."""

import json
import pytest
from pydantic import ValidationError

from khainglens.models import (
    AnalysisConfig,
    AnalysisScope,
    Anomaly,
    AnomalyType,
    DuplicateGroup,
    InputFormat,
    LoadResult,
    MalformedRecord,
    ParsedRecord,
    Severity,
    build_excerpt,
)


def test_default_analysis_config_values() -> None:
    """Verify default configuration matches approved architectural specifications."""
    config = AnalysisConfig()
    assert config.iqr_multiplier == 1.5
    assert config.min_records_for_iqr == 4
    assert config.repeated_space_threshold == 2
    assert config.detect_normalized_duplicates is True
    assert config.flag_internal_newlines is True
    assert config.flag_tabs is True
    assert config.max_records_limit == 100_000
    assert config.excerpt_max_chars == 300


def test_analysis_config_validation_bounds() -> None:
    """Verify invalid configuration parameters raise Pydantic ValidationError."""
    # Valid custom configuration up to 500,000 limit
    valid_custom = AnalysisConfig(iqr_multiplier=2.5, max_records_limit=500_000)
    assert valid_custom.max_records_limit == 500_000

    # Invalid IQR multiplier (< 0.5)
    with pytest.raises(ValidationError):
        AnalysisConfig(iqr_multiplier=0.1)

    # Invalid max_records_limit (> 500,000)
    with pytest.raises(ValidationError):
        AnalysisConfig(max_records_limit=600_000)

    # Invalid repeated_space_threshold (< 2)
    with pytest.raises(ValidationError):
        AnalysisConfig(repeated_space_threshold=1)


def test_parsed_record_immutability_and_burmese_preservation() -> None:
    """Verify ParsedRecord preserves Burmese text exactly and prevents mutation."""
    burmese_sample = "မြန်မာဘာသာစကား အချက်အလက် စစ်ဆေးခြင်း"
    record = ParsedRecord(
        record_index=0,
        source_line_number=1,
        raw_text=burmese_sample,
        is_missing_value=False,
    )
    assert record.raw_text == burmese_sample
    assert record.is_missing_value is False

    # Attempting to modify raw_text must fail due to frozen=True
    with pytest.raises(ValidationError):
        record.raw_text = "modified text"  # type: ignore[misc]


def test_missing_vs_empty_record_representation() -> None:
    """Verify missing values (None) and empty strings ('') are distinct in ParsedRecord."""
    missing_rec = ParsedRecord(
        record_index=0, source_line_number=2, raw_text=None, is_missing_value=True
    )
    empty_rec = ParsedRecord(
        record_index=1, source_line_number=3, raw_text="", is_missing_value=False
    )

    assert missing_rec.raw_text is None
    assert missing_rec.is_missing_value is True
    assert empty_rec.raw_text == ""
    assert empty_rec.is_missing_value is False


def test_consolidated_anomaly_schema_and_serialization() -> None:
    """Verify consolidated Anomaly model stores occurrence counts, positions, and serializes to JSON."""
    anomaly = Anomaly(
        anomaly_id="anom_000005_zero_width_character",
        record_index=5,
        source_line_number=6,
        issue_type=AnomalyType.ZERO_WIDTH_CHARACTER,
        severity=Severity.WARNING,
        title="Zero-Width Characters Detected (3 occurrences)",
        explanation="Record contains 3 invisible zero-width characters.",
        original_excerpt="မြန်\u200bမာ\u200bစာ\u200c",
        evidence="3 zero-width characters at positions [4, 7, 10]",
        suggested_action="Inspect whether U+200B is intentional syllable segmentation.",
        occurrence_count=3,
        char_position=4,
        char_positions=[4, 7, 10],
        metadata={"code_points": ["U+200B", "U+200B", "U+200C"]},
    )

    assert anomaly.occurrence_count == 3
    assert anomaly.char_position == 4
    assert anomaly.char_positions == [4, 7, 10]
    assert anomaly.issue_type == AnomalyType.ZERO_WIDTH_CHARACTER
    assert anomaly.severity == Severity.WARNING

    # Verify clean JSON round-trip with Burmese Unicode preserved
    json_str = anomaly.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["occurrence_count"] == 3
    assert parsed["char_positions"] == [4, 7, 10]
    assert parsed["issue_type"] == "zero_width_character"
    assert "မြန်" in parsed["original_excerpt"]


def test_duplicate_group_validation() -> None:
    """Verify DuplicateGroup enforces minimum occurrence_count >= 2."""
    group = DuplicateGroup(
        group_id=1,
        representative_excerpt="မင်္ဂလာပါ",
        record_indices=[2, 9, 14],
        occurrence_count=3,
        surplus_count=2,
    )
    assert group.comparison_basis == "raw_exact_match"
    assert group.occurrence_count == 3
    assert group.surplus_count == 2

    with pytest.raises(ValidationError):
        DuplicateGroup(
            group_id=2,
            representative_excerpt="Single item",
            record_indices=[1],
            occurrence_count=1,
            surplus_count=0,
        )


def test_load_result_truncation_metadata() -> None:
    """Verify LoadResult tracks truncated_subset scope and malformed records."""
    malformed = MalformedRecord(
        record_index=1,
        source_line_number=2,
        raw_excerpt='{"text": unclosed',
        reason="Invalid JSON syntax",
        error_detail="Expecting value: line 1 column 10",
    )
    load_res = LoadResult(
        filename="large.jsonl",
        file_format=InputFormat.JSONL,
        encoding_used="utf-8",
        has_bom=False,
        selected_field="text",
        analysis_scope=AnalysisScope.TRUNCATED_SUBSET,
        truncated_at_limit=True,
        max_records_limit_applied=100_000,
        analyzed_raw_records=1,
        valid_records=[],
        malformed_records=[malformed],
    )
    assert load_res.truncated_at_limit is True
    assert load_res.analysis_scope == AnalysisScope.TRUNCATED_SUBSET
    assert len(load_res.malformed_records) == 1


def test_build_excerpt_short_and_null() -> None:
    """Verify build_excerpt handles None, empty strings, and short texts without truncation."""
    assert build_excerpt(None) == "<MISSING / NULL>"
    assert build_excerpt("") == ""
    assert build_excerpt("Short Burmese: မြန်မာ", max_chars=50) == "Short Burmese: မြန်မာ"


def test_build_excerpt_long_and_focused() -> None:
    """Verify build_excerpt truncates long strings and centers around focus_pos when given."""
    long_text = "A" * 100 + "TARGET" + "B" * 100
    unfocused = build_excerpt(long_text, max_chars=50)
    assert len(unfocused) == 50
    assert unfocused.endswith("...")

    focused = build_excerpt(long_text, max_chars=60, focus_pos=102)
    assert "TARGET" in focused
    assert focused.startswith("...")
    assert focused.endswith("...")