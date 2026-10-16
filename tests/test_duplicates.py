"""Unit tests for khainglens.duplicate_detection covering exact and NFC-normalized duplicates."""

from khainglens.duplicate_detection import detect_duplicates
from khainglens.models import AnalysisConfig, AnomalyType, ParsedRecord, Severity


def test_no_duplicates_returns_empty() -> None:
    """Verify a dataset with all unique records produces 0 duplicate groups and 0 anomalies."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="First unique sentence"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="မင်္ဂလာပါ မြန်မာစာ"),
        ParsedRecord(record_index=2, source_line_number=3, raw_text="Third unique sentence"),
    ]
    groups, anomalies, norm_count = detect_duplicates(records)
    assert groups == []
    assert anomalies == []
    assert norm_count == 0


def test_blank_empty_and_missing_records_excluded_from_duplicates() -> None:
    """Verify Decision 1: None, '', and whitespace-only strings are never grouped as duplicates."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text=None, is_missing_value=True),
        ParsedRecord(record_index=1, source_line_number=2, raw_text=None, is_missing_value=True),
        ParsedRecord(record_index=2, source_line_number=3, raw_text=""),
        ParsedRecord(record_index=3, source_line_number=4, raw_text=""),
        ParsedRecord(record_index=4, source_line_number=5, raw_text=""),
        ParsedRecord(record_index=5, source_line_number=6, raw_text="   "),
        ParsedRecord(record_index=6, source_line_number=7, raw_text="   "),
        ParsedRecord(record_index=7, source_line_number=8, raw_text="\t\n"),
        ParsedRecord(record_index=8, source_line_number=9, raw_text="\t\n"),
    ]
    groups, anomalies, norm_count = detect_duplicates(records)
    assert groups == []
    assert anomalies == []
    assert norm_count == 0


def test_multiple_duplicate_groups_and_three_or_more_identical_records() -> None:
    """Verify group counts, surplus counts, and affected record indices for 2-item and 3-item groups."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="မင်္ဂလာပါ"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="Unique item"),
        ParsedRecord(record_index=2, source_line_number=3, raw_text="မင်္ဂလာပါ"),
        ParsedRecord(record_index=3, source_line_number=4, raw_text="Repeated English"),
        ParsedRecord(record_index=4, source_line_number=5, raw_text="မင်္ဂလာပါ"),
        ParsedRecord(record_index=5, source_line_number=6, raw_text="Repeated English"),
    ]

    groups, anomalies, norm_count = detect_duplicates(records)

    assert len(groups) == 2
    assert norm_count == 0

    # Group 1: "မင်္ဂလာပါ" at indices [0, 2, 4] -> occurrence_count=3, surplus_count=2
    g1 = next(g for g in groups if g.representative_excerpt == "မင်္ဂလာပါ")
    assert g1.record_indices == [0, 2, 4]
    assert g1.occurrence_count == 3
    assert g1.surplus_count == 2

    # Group 2: "Repeated English" at indices [3, 5] -> occurrence_count=2, surplus_count=1
    g2 = next(g for g in groups if g.representative_excerpt == "Repeated English")
    assert g2.record_indices == [3, 5]
    assert g2.occurrence_count == 2
    assert g2.surplus_count == 1

    # Total affected records = 3 + 2 = 5
    exact_anoms = [a for a in anomalies if a.issue_type == AnomalyType.EXACT_DUPLICATE]
    assert len(exact_anoms) == 5
    assert [a.record_index for a in exact_anoms] == [0, 2, 3, 4, 5]
    assert all(a.severity == Severity.INFO for a in exact_anoms)
    # Each record's EXACT_DUPLICATE anomaly has occurrence_count == 1 so sums equal 5
    assert sum(a.occurrence_count for a in exact_anoms) == 5


def test_raw_exact_duplicate_does_not_merge_whitespace_variants() -> None:
    """Verify exact duplicate detection compares raw text without stripping spaces."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="မင်္ဂလာပါ"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="မင်္ဂလာပါ "),
    ]
    groups, anomalies, _ = detect_duplicates(records)
    assert groups == []
    assert anomalies == []


def test_burmese_normalized_duplicates_flagged_separately_from_exact_duplicates() -> None:
    """Verify visually identical Burmese strings with different combining orders trigger NORMALIZED_DUPLICATE."""
    # U+1000 (က) + U+1037 (့ dot below, CCC=7) + U+103A (် asat, CCC=9) -> canonical NFC order
    canonical_burmese = "\u1000\u1037\u103a"
    # U+1000 (က) + U+103A (် asat, CCC=9) + U+1037 (့ dot below, CCC=7) -> non-NFC order
    non_canonical_burmese = "\u1000\u103a\u1037"

    assert canonical_burmese != non_canonical_burmese

    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text=canonical_burmese),
        ParsedRecord(record_index=1, source_line_number=2, raw_text=non_canonical_burmese),
    ]

    groups, anomalies, norm_count = detect_duplicates(records)

    # Not exact duplicates because raw code points differ!
    assert groups == []
    assert norm_count == 1

    # Both records receive NORMALIZED_DUPLICATE warnings
    assert len(anomalies) == 2
    assert all(a.issue_type == AnomalyType.NORMALIZED_DUPLICATE for a in anomalies)
    assert all(a.severity == Severity.WARNING for a in anomalies)
    assert [a.record_index for a in anomalies] == [0, 1]


def test_latin_precomposed_vs_decomposed_normalized_duplicate() -> None:
    """Verify Latin NFC ('é' U+00E9) vs NFD ('e' + U+0301) triggers NORMALIZED_DUPLICATE."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="caf\u00e9"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="cafe\u0301"),
    ]
    groups, anomalies, norm_count = detect_duplicates(records)
    assert groups == []
    assert norm_count == 1
    assert len(anomalies) == 2
    assert all(a.issue_type == AnomalyType.NORMALIZED_DUPLICATE for a in anomalies)


def test_disable_normalized_duplicates_in_config() -> None:
    """Verify detect_normalized_duplicates=False disables the NFC comparison pass."""
    records = [
        ParsedRecord(record_index=0, source_line_number=1, raw_text="\u1000\u1037\u103a"),
        ParsedRecord(record_index=1, source_line_number=2, raw_text="\u1000\u103a\u1037"),
    ]
    cfg = AnalysisConfig(detect_normalized_duplicates=False)
    groups, anomalies, norm_count = detect_duplicates(records, config=cfg)
    assert groups == []
    assert anomalies == []
    assert norm_count == 0