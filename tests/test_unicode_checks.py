"""Unit tests for khainglens.unicode_checks covering Unicode anomalies and Burmese script rules."""

from khainglens.models import AnomalyType, ParsedRecord, Severity
from khainglens.unicode_checks import (
    check_record_unicode,
    inspect_text_code_points,
    visualize_invisible_characters,
)


def test_valid_burmese_unicode_sequences_never_falsely_flagged() -> None:
    """CRITICAL: Verify valid Myanmar Unicode sequences (Kinzi, medials, stacked consonants, U+1031) produce 0 anomalies."""
    valid_burmese_samples = [
        # 1. Kinzi sequence (U+1004 U+103A U+1039 + base consonant) + medials + tones
        "မင်္ဂလာပါခင်ဗျာ။ မြန်မာဘာသာစကား အချက်အလက်များကို စစ်ဆေးခြင်းဖြစ်ပါသည်။",
        # 2. Stacked consonants (ပတ်သက်, တက္ကသိုလ်, အန္တရာယ်, ကမ္ဘာ)
        "ရန်ကုန်တက္ကသိုလ်နှင့် မန္တလေးတက္ကသိုလ်တို့တွင် သုတေသနပြုလုပ်ကြသည်။",
        # 3. Complex medials (ကြွေးမြီ, လွှတ်တော်, ချွေး, မြွေ) and logical-order U+1031
        "ပြည်ထောင်စုလွှတ်တော်တွင် ကြွေးမြီဆိုင်ရာ ဥပဒေကြမ်းကို ဆွေးနွေးခဲ့သည်။",
        # 4. Independent vowels (ဣ, ဤ, ဥ, ဦ, ဧ, ဩ, ဪ), Great Sa (ဿ), digits, and symbols (၌, ၍, ၎င်း, ၏)
        "ဤစာတမ်း၌ ၂၀၂၆ ခုနှစ်အတွင်း ၎င်းတို့၏ ဦးဆောင်မှုဖြင့် အောင်မြင်ခဲ့ကြောင်း ဖော်ပြထားသည်။",
    ]

    for idx, text in enumerate(valid_burmese_samples):
        rec = ParsedRecord(record_index=idx, source_line_number=idx + 1, raw_text=text)
        findings = check_record_unicode(rec)
        assert findings == [], f"False positive on valid Burmese sample #{idx + 1}: {findings}"


def test_zero_width_characters_consolidated_within_record() -> None:
    """Verify multiple zero-width characters in one record produce ONE consolidated WARNING."""
    # Contains two U+200B (ZWSP) and one U+200C (ZWNJ)
    raw = "မြန်\u200bမာ\u200bစာ\u200cနမူနာ"
    rec = ParsedRecord(record_index=3, source_line_number=4, raw_text=raw)

    findings = check_record_unicode(rec)
    assert len(findings) == 1
    anom = findings[0]
    assert anom.issue_type == AnomalyType.ZERO_WIDTH_CHARACTER
    assert anom.severity == Severity.WARNING
    assert anom.occurrence_count == 3
    assert len(anom.char_positions) == 3
    assert "ZWSP" in anom.original_excerpt
    assert rec.raw_text == raw  # Unmutated


def test_embedded_bom_detection() -> None:
    """Verify U+FEFF inside a record is flagged as EMBEDDED_BOM with WARNING severity."""
    rec = ParsedRecord(
        record_index=0,
        source_line_number=1,
        raw_text="First part \ufeff second part \ufeff end",
    )
    findings = check_record_unicode(rec)
    assert len(findings) == 1
    assert findings[0].issue_type == AnomalyType.EMBEDDED_BOM
    assert findings[0].severity == Severity.WARNING
    assert findings[0].occurrence_count == 2


def test_unexpected_control_characters_and_replacement_char() -> None:
    """Verify C0/C1 control characters and U+FFFD are flagged as UNEXPECTED_CONTROL_CHAR (ERROR)."""
    rec = ParsedRecord(
        record_index=1,
        source_line_number=2,
        raw_text="Corrupted\x00bell\x07and\ufffdreplacement",
    )
    findings = check_record_unicode(rec)
    assert len(findings) == 1
    anom = findings[0]
    assert anom.issue_type == AnomalyType.UNEXPECTED_CONTROL_CHAR
    assert anom.severity == Severity.ERROR
    assert anom.occurrence_count == 3


def test_unusual_unicode_spaces_detected_as_info() -> None:
    """Verify NBSP (U+00A0) and Ideographic Space (U+3000) are flagged as UNUSUAL_UNICODE_SPACE (INFO)."""
    rec = ParsedRecord(
        record_index=2,
        source_line_number=3,
        raw_text="Word1\u00a0Word2\u3000Word3",
    )
    findings = check_record_unicode(rec)
    assert len(findings) == 1
    anom = findings[0]
    assert anom.issue_type == AnomalyType.UNUSUAL_UNICODE_SPACE
    assert anom.severity == Severity.INFO
    assert anom.occurrence_count == 2
    assert anom.char_positions == [5, 11]


def test_unicode_normalization_diff_detected_for_non_nfc_burmese() -> None:
    """Verify non-NFC combining order (U+103A before U+1037) triggers UNICODE_NORMALIZATION_DIFF."""
    # U+1000 (က) + U+103A (asat, CCC=9) + U+1037 (dot below, CCC=7) -> non-NFC
    non_nfc = "\u1000\u103a\u1037"
    rec = ParsedRecord(record_index=5, source_line_number=6, raw_text=non_nfc)

    findings = check_record_unicode(rec)
    types = [f.issue_type for f in findings]
    assert AnomalyType.UNICODE_NORMALIZATION_DIFF in types
    norm_anom = next(f for f in findings if f.issue_type == AnomalyType.UNICODE_NORMALIZATION_DIFF)
    assert norm_anom.severity == Severity.INFO


def test_myanmar_orphan_combining_mark_detected() -> None:
    """Verify a Myanmar vowel/combining mark at index 0 or after space triggers MYANMAR_ORPHAN_COMBINING."""
    # Starts directly with dependent vowel U+102C (ာ) without a base consonant
    rec = ParsedRecord(record_index=0, source_line_number=1, raw_text="\u102cမင်္ဂလာပါ")
    findings = check_record_unicode(rec)
    types = [f.issue_type for f in findings]
    assert AnomalyType.MYANMAR_ORPHAN_COMBINING in types


def test_myanmar_duplicate_adjacent_diacritic_detected() -> None:
    """Verify doubled Myanmar vowel or tone marks (e.g. U+1037 U+1037 or U+102D U+102D) are flagged."""
    # က + ိ + ိ (double vowel i) and န + ့ + ့ (double dot below)
    raw = "က\u102d\u102dန\u1037\u1037"
    rec = ParsedRecord(record_index=8, source_line_number=9, raw_text=raw)

    findings = check_record_unicode(rec)
    dup_anoms = [f for f in findings if f.issue_type == AnomalyType.MYANMAR_DUPLICATE_DIACRITIC]
    assert len(dup_anoms) == 1
    assert dup_anoms[0].severity == Severity.WARNING
    assert dup_anoms[0].occurrence_count == 2


def test_myanmar_suspected_zawgyi_visual_order_and_repurposed_codepoints() -> None:
    """Verify Zawgyi visual-order U+1031 and Zawgyi medial code points trigger MYANMAR_SUSPECTED_ZAWGYI."""
    # Zawgyi typing for 'မေ': U+1031 (ေ) typed BEFORE U+1019 (မ), plus U+1033 (Zawgyi medial)
    zawgyi_sample = "\u1031\u1019\u1000\u1033"
    rec = ParsedRecord(record_index=9, source_line_number=10, raw_text=zawgyi_sample)

    findings = check_record_unicode(rec)
    types = {f.issue_type for f in findings}
    assert AnomalyType.MYANMAR_SUSPECTED_ZAWGYI in types
    z_anom = next(f for f in findings if f.issue_type == AnomalyType.MYANMAR_SUSPECTED_ZAWGYI)
    assert z_anom.severity == Severity.WARNING
    assert "Mon" in z_anom.explanation  # Explains ethnic script nuance


def test_visualize_invisible_characters_and_code_point_inspector() -> None:
    """Verify helper functions render invisible characters and code point tables accurately."""
    sample = "က\u200bခ\ufeff"
    vis = visualize_invisible_characters(sample)
    assert "[ZWSP U+200B]" in vis
    assert "[BOM U+FEFF]" in vis

    table = inspect_text_code_points(sample)
    assert len(table) == 4
    assert table[0]["code_point"] == "U+1000"
    assert table[0]["is_special_or_invisible"] is False
    assert table[1]["code_point"] == "U+200B"
    assert table[1]["is_special_or_invisible"] is True


def test_empty_or_missing_records_skipped_cleanly_by_unicode_checks() -> None:
    """Verify empty and missing records return [] in check_record_unicode (handled by whitespace_checks)."""
    missing_rec = ParsedRecord(record_index=0, source_line_number=1, raw_text=None, is_missing_value=True)
    empty_rec = ParsedRecord(record_index=1, source_line_number=2, raw_text="")
    assert check_record_unicode(missing_rec) == []
    assert check_record_unicode(empty_rec) == []