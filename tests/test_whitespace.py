"""Unit tests for khainglens.whitespace_checks covering missing, empty, whitespace-only, and anomalies."""

from khainglens.models import AnalysisConfig, AnomalyType, ParsedRecord, Severity
from khainglens.whitespace_checks import check_record_whitespace


def test_missing_empty_and_whitespace_only_distinction() -> None:
    """Verify None, '', and whitespace-only strings produce distinct AnomalyTypes and severities."""
    missing_rec = ParsedRecord(
        record_index=0, source_line_number=1, raw_text=None, is_missing_value=True
    )
    empty_rec = ParsedRecord(
        record_index=1, source_line_number=2, raw_text="", is_missing_value=False
    )
    spaces_only_rec = ParsedRecord(
        record_index=2, source_line_number=3, raw_text="    ", is_missing_value=False
    )
    tabs_newlines_rec = ParsedRecord(
        record_index=3, source_line_number=4, raw_text="\t\r\n ", is_missing_value=False
    )

    m_anom = check_record_whitespace(missing_rec)
    assert len(m_anom) == 1
    assert m_anom[0].issue_type == AnomalyType.MISSING_TEXT_VALUE
    assert m_anom[0].severity == Severity.ERROR

    e_anom = check_record_whitespace(empty_rec)
    assert len(e_anom) == 1
    assert e_anom[0].issue_type == AnomalyType.EMPTY_STRING
    assert e_anom[0].severity == Severity.WARNING

    s_anom = check_record_whitespace(spaces_only_rec)
    assert len(s_anom) == 1
    assert s_anom[0].issue_type == AnomalyType.WHITESPACE_ONLY
    assert s_anom[0].severity == Severity.WARNING
    assert s_anom[0].occurrence_count == 4

    tn_anom = check_record_whitespace(tabs_newlines_rec)
    assert len(tn_anom) == 1
    assert tn_anom[0].issue_type == AnomalyType.WHITESPACE_ONLY
    assert tn_anom[0].occurrence_count == 4
    assert "TAB" in tn_anom[0].evidence


def test_clean_english_and_burmese_records_produce_no_whitespace_anomalies() -> None:
    """Verify clean English and Burmese sentences with normal single spaces are not flagged."""
    clean_en = ParsedRecord(
        record_index=0,
        source_line_number=1,
        raw_text="This is a clean English sentence with single spaces.",
    )
    clean_my = ParsedRecord(
        record_index=1,
        source_line_number=2,
        raw_text="မြန်မာဘာသာစကား အချက်အလက်များကို စနစ်တကျ စစ်ဆေးပါသည်။",
    )

    assert check_record_whitespace(clean_en) == []
    assert check_record_whitespace(clean_my) == []


def test_leading_and_trailing_whitespace_detection() -> None:
    """Verify leading and trailing whitespace are detected accurately without mutating raw_text."""
    original = "   မင်္ဂလာပါ  "
    rec = ParsedRecord(record_index=4, source_line_number=5, raw_text=original)

    findings = check_record_whitespace(rec)
    issue_types = [f.issue_type for f in findings]

    assert AnomalyType.LEADING_WHITESPACE in issue_types
    assert AnomalyType.TRAILING_WHITESPACE in issue_types
    # Leading/trailing spaces must not be double-counted as internal REPEATED_SPACES
    assert AnomalyType.REPEATED_SPACES not in issue_types

    leading = next(f for f in findings if f.issue_type == AnomalyType.LEADING_WHITESPACE)
    trailing = next(f for f in findings if f.issue_type == AnomalyType.TRAILING_WHITESPACE)

    assert leading.occurrence_count == 3
    assert leading.char_positions == [0, 1, 2]
    assert trailing.occurrence_count == 2
    assert trailing.char_position == len(original) - 2
    assert rec.raw_text == original


def test_repeated_spaces_consolidation_within_single_record() -> None:
    """Verify multiple spans of repeated spaces in one record consolidate into ONE Anomaly."""
    rec = ParsedRecord(
        record_index=7,
        source_line_number=8,
        raw_text="First  second   third    fourth",
    )
    findings = check_record_whitespace(rec)

    assert len(findings) == 1
    anom = findings[0]
    assert anom.issue_type == AnomalyType.REPEATED_SPACES
    assert anom.severity == Severity.INFO
    assert anom.occurrence_count == 3
    assert len(anom.char_positions) == 3
    assert anom.metadata["span_lengths"] == [2, 3, 4]


def test_repeated_space_custom_threshold() -> None:
    """Verify repeated_space_threshold configuration controls minimum consecutive space count."""
    rec = ParsedRecord(
        record_index=0, source_line_number=1, raw_text="Two  spaces vs three   spaces"
    )
    config_thresh_3 = AnalysisConfig(repeated_space_threshold=3)

    findings = check_record_whitespace(rec, config=config_thresh_3)
    assert len(findings) == 1
    assert findings[0].occurrence_count == 1
    assert findings[0].metadata["span_lengths"] == [3]


def test_embedded_tabs_and_newlines_consolidated() -> None:
    """Verify embedded tabs and newlines inside a non-empty record produce one consolidated finding."""
    rec = ParsedRecord(
        record_index=2,
        source_line_number=3,
        raw_text="Col1\tCol2\nLine2\r\nLine3",
    )
    findings = check_record_whitespace(rec)

    assert len(findings) == 1
    anom = findings[0]
    assert anom.issue_type == AnomalyType.UNUSUAL_TAB_OR_NEWLINE
    assert anom.severity == Severity.WARNING
    assert anom.occurrence_count == 4  # \t, \n, \r, \n
    assert len(anom.char_positions) == 4


def test_disable_tab_and_newline_flags_via_config() -> None:
    """Verify flag_tabs=False and flag_internal_newlines=False suppress those checks when configured."""
    rec = ParsedRecord(record_index=0, source_line_number=1, raw_text="Hello\tWorld\nNext")
    cfg = AnalysisConfig(flag_tabs=False, flag_internal_newlines=False)
    assert check_record_whitespace(rec, config=cfg) == []


def test_multiple_distinct_whitespace_anomaly_types_on_same_record() -> None:
    """Verify a record with leading space, repeated internal space, and tab gets 3 separate findings."""
    rec = ParsedRecord(
        record_index=10,
        source_line_number=11,
        raw_text="  မြန်မာ  စာ\tနမူနာ",
    )
    findings = check_record_whitespace(rec)
    types = {f.issue_type for f in findings}
    assert types == {
        AnomalyType.LEADING_WHITESPACE,
        AnomalyType.REPEATED_SPACES,
        AnomalyType.UNUSUAL_TAB_OR_NEWLINE,
    }