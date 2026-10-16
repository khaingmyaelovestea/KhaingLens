"""End-to-end integration tests for khainglens.analyzer pipeline."""

from khainglens.analyzer import (
    analyze_dataset,
    build_record_lookup,
    inspect_dataset_bytes,
)
from khainglens.loader import load_dataset
from khainglens.models import (
    AnalysisConfig,
    AnalysisScope,
    AnomalyType,
    MalformedRecord,
    ParsedRecord,
    Severity,
)


def test_complete_pipeline_jsonl_burmese_and_english_integration() -> None:
    """Verify full pipeline across malformed lines, duplicates, whitespace, Burmese Unicode, and stats."""
    jsonl_lines = [
        # Record 0 (Line 1): Valid Burmese with Kinzi (clean, duplicated at Record 1)
        '{"id": 1, "text": "မင်္ဂလာပါ မြန်မာဘာသာစကား စစ်ဆေးရေး ကိရိယာ"}',
        # Record 1 (Line 2): Exact duplicate of Record 0
        '{"id": 2, "text": "မင်္ဂလာပါ မြန်မာဘာသာစကား စစ်ဆေးရေး ကိရိယာ"}',
        # Record 2 (Line 3): Leading/trailing whitespace + 2 ZWSP + repeated internal spaces
        '{"id": 3, "text": "  မြန်\\u200bမာ\\u200bစာ   နမူနာ  "}',
        # Record 3 (Line 4): Empty string
        '{"id": 4, "text": ""}',
        # Record 4 (Line 5): Null / missing text field
        '{"id": 5, "text": null}',
        # Record 5 (Line 6): Malformed JSON line
        '{"id": 6, "text": broken_json_syntax}',
        # Record 6 (Line 7): Clean English sentence
        '{"id": 7, "text": "Clean English reference sentence for dataset inspection."}',
        # Record 7 (Line 8): Suspected Zawgyi visual-order U+1031 before consonant
        '{"id": 8, "text": "\\u1031\\u1019\\u1010\\u1039\\u1010\\u102c"}',
    ]
    raw_bytes = "\n".join(jsonl_lines).encode("utf-8")

    result = inspect_dataset_bytes(raw_bytes, filename="integration.jsonl", selected_field="text")

    # 1. Verify record counts
    stats = result.stats
    assert stats.analyzed_raw_records == 8
    assert stats.successfully_parsed_records == 7
    assert stats.malformed_record_count == 1
    assert stats.empty_string_record_count == 1
    assert stats.missing_value_record_count == 1
    assert stats.non_empty_analyzable_records == 5

    # 2. Verify duplicate group (Records 0 and 1)
    assert stats.duplicate_group_count == 1
    assert stats.total_records_in_duplicate_groups == 2
    assert stats.surplus_duplicate_occurrences == 1
    assert result.duplicate_groups[0].record_indices == [0, 1]

    # 3. Verify Record 2 has multiple distinct anomalies consolidated per type
    rec2_anoms = [a for a in result.anomalies if a.record_index == 2]
    rec2_types = {a.issue_type for a in rec2_anoms}
    assert AnomalyType.LEADING_WHITESPACE in rec2_types
    assert AnomalyType.TRAILING_WHITESPACE in rec2_types
    assert AnomalyType.REPEATED_SPACES in rec2_types
    assert AnomalyType.ZERO_WIDTH_CHARACTER in rec2_types

    zw_anom = next(a for a in rec2_anoms if a.issue_type == AnomalyType.ZERO_WIDTH_CHARACTER)
    assert zw_anom.occurrence_count == 2

    # 4. Verify Malformed Record 5 was converted to MALFORMED_RECORD (ERROR)
    rec5_anoms = [a for a in result.anomalies if a.record_index == 5]
    assert len(rec5_anoms) == 1
    assert rec5_anoms[0].issue_type == AnomalyType.MALFORMED_RECORD
    assert rec5_anoms[0].severity == Severity.ERROR

    # 5. Verify Clean Record 6 has 0 anomalies
    rec6_anoms = [a for a in result.anomalies if a.record_index == 6]
    assert rec6_anoms == []

    # 6. Verify Record 7 triggered MYANMAR_SUSPECTED_ZAWGYI
    rec7_types = {a.issue_type for a in result.anomalies if a.record_index == 7}
    assert AnomalyType.MYANMAR_SUSPECTED_ZAWGYI in rec7_types


def test_original_input_records_never_mutated_by_analyzer() -> None:
    """Verify no module in the pipeline alters or normalizes raw_text."""
    raw_texts = [
        "  Leading and trailing  ",
        "မြန်\u200bမာ\u1000\u103a\u1037",
        "",
        "   ",
    ]
    txt_bytes = "\n".join(raw_texts).encode("utf-8")
    load_res = load_dataset(txt_bytes, filename="immutable.txt")
    snapshot_before = [r.raw_text for r in load_res.valid_records]

    result = analyze_dataset(load_res)
    snapshot_after = [r.raw_text for r in result.load_result.valid_records]

    assert snapshot_after == snapshot_before
    assert snapshot_after == raw_texts


def test_consolidation_invariant_at_most_one_anomaly_per_record_and_type() -> None:
    """Verify no (record_index, issue_type) pair ever appears twice in result.anomalies."""
    noisy_line = "   Word1  Word2   Word3\u200b\u200b\u200c\t\t\ufeff\ufeff   "
    txt_bytes = f"{noisy_line}\n{noisy_line}\n".encode("utf-8")

    result = inspect_dataset_bytes(txt_bytes, filename="noisy.txt")
    seen_pairs: set[tuple[int, AnomalyType]] = set()
    for anom in result.anomalies:
        pair = (anom.record_index, anom.issue_type)
        assert pair not in seen_pairs, f"Duplicate anomaly emitted for {pair}"
        seen_pairs.add(pair)


def test_build_record_lookup_handles_interleaved_malformed_and_valid_records() -> None:
    """Verify build_record_lookup retrieves exact ParsedRecord or MalformedRecord by record_index."""
    csv_bytes = (
        "id,text\n"
        "1,First valid\n"
        "2,Malformed,Extra,Col\n"
        "3,Third valid\n"
    ).encode("utf-8")

    result = inspect_dataset_bytes(csv_bytes, filename="interleaved.csv", csv_has_header=True)
    lookup = build_record_lookup(result.load_result)

    assert isinstance(lookup[0], ParsedRecord)
    assert lookup[0].raw_text == "First valid"

    assert isinstance(lookup[1], MalformedRecord)
    assert lookup[1].source_line_number == 3

    assert isinstance(lookup[2], ParsedRecord)
    assert lookup[2].raw_text == "Third valid"


def test_truncated_analysis_pipeline_propagates_scope() -> None:
    """Verify max_records_limit truncation propagates through analyze_dataset into stats."""
    txt_bytes = "\n".join([f"Sentence number {i}" for i in range(40)]).encode("utf-8")
    cfg = AnalysisConfig(max_records_limit=12)

    result = inspect_dataset_bytes(txt_bytes, filename="forty.txt", config=cfg)

    assert result.load_result.truncated_at_limit is True
    assert result.stats.truncated_at_limit is True
    assert result.stats.analysis_scope == AnalysisScope.TRUNCATED_SUBSET
    assert result.stats.analyzed_raw_records == 12


def test_csv_header_toggle_through_inspect_dataset_bytes() -> None:
    """Verify inspect_dataset_bytes respects csv_has_header=True vs False end-to-end."""
    csv_bytes = "col_a,col_b\nval1,val2\n".encode("utf-8")

    res_hdr = inspect_dataset_bytes(csv_bytes, "toggle.csv", csv_has_header=True)
    assert res_hdr.stats.analyzed_raw_records == 1

    res_no_hdr = inspect_dataset_bytes(csv_bytes, "toggle.csv", csv_has_header=False)
    assert res_no_hdr.stats.analyzed_raw_records == 2