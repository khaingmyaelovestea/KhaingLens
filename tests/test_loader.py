"""Unit tests for khainglens.loader covering CSV, JSONL, TXT, BOM, headers, and truncation."""

import pytest

from khainglens.loader import inspect_file_schema, load_dataset
from khainglens.models import (
    AnalysisConfig,
    AnalysisScope,
    FileEncodingError,
    InputFormat,
    UnsupportedFormatError,
)


def test_load_valid_txt_and_burmese_preservation() -> None:
    """Verify TXT loader preserves exact line strings including leading/trailing spaces and Burmese."""
    content = (
        "မင်္ဂလာပါ မြန်မာစာ\n"
        "  Leading and trailing spaces  \n"
        "\n"
        "Final line without trailing newline"
    ).encode("utf-8")

    result = load_dataset(content, filename="corpus.txt")

    assert result.file_format == InputFormat.TXT
    assert result.has_bom is False
    assert result.analysis_scope == AnalysisScope.COMPLETE
    assert result.truncated_at_limit is False
    assert result.analyzed_raw_records == 4
    assert len(result.valid_records) == 4
    assert len(result.malformed_records) == 0

    assert result.valid_records[0].raw_text == "မင်္ဂလာပါ မြန်မာစာ"
    assert result.valid_records[0].source_line_number == 1
    assert result.valid_records[1].raw_text == "  Leading and trailing spaces  "
    assert result.valid_records[2].raw_text == ""
    assert result.valid_records[3].source_line_number == 4


def test_utf8_bom_stripped_from_header_but_internal_bom_preserved() -> None:
    """Verify file-initial UTF-8 BOM is stripped while internal U+FEFF remains for Unicode checks."""
    # Starts with UTF-8 BOM (\xef\xbb\xbf), and Line 2 has an internal U+FEFF
    raw_bytes = b"\xef\xbb\xbfFirst line\nSecond \xef\xbb\xbfline"
    result = load_dataset(raw_bytes, filename="bom_test.txt")

    assert result.has_bom is True
    assert result.encoding_used == "utf-8-sig"
    assert result.valid_records[0].raw_text == "First line"
    assert "\ufeff" not in (result.valid_records[0].raw_text or "")
    assert result.valid_records[1].raw_text == "Second \ufeffline"


def test_csv_header_vs_headerless_reparse() -> None:
    """Verify toggling csv_has_header re-parses Row 1 as data and generates col_0, col_1 names."""
    csv_bytes = (
        "id,text\n"
        "1,မြန်မာစာ နမူနာ ၁\n"
        "2,မြန်မာစာ နမူနာ ၂\n"
    ).encode("utf-8")

    # 1. Default: csv_has_header=True
    schema_with_hdr = inspect_file_schema(csv_bytes, "data.csv", csv_has_header=True)
    assert schema_with_hdr.candidate_fields == ["id", "text"]
    assert schema_with_hdr.suggested_field == "text"

    res_with_hdr = load_dataset(csv_bytes, "data.csv", csv_has_header=True)
    assert res_with_hdr.csv_has_header is True
    assert res_with_hdr.selected_field == "text"
    assert len(res_with_hdr.valid_records) == 2
    assert res_with_hdr.valid_records[0].record_index == 0
    assert res_with_hdr.valid_records[0].source_line_number == 2
    assert res_with_hdr.valid_records[0].raw_text == "မြန်မာစာ နမူနာ ၁"

    # 2. Unchecked: csv_has_header=False
    schema_no_hdr = inspect_file_schema(csv_bytes, "data.csv", csv_has_header=False)
    assert schema_no_hdr.candidate_fields == ["col_0", "col_1"]

    res_no_hdr = load_dataset(
        csv_bytes, "data.csv", selected_field="col_1", csv_has_header=False
    )
    assert res_no_hdr.csv_has_header is False
    assert res_no_hdr.selected_field == "col_1"
    assert len(res_no_hdr.valid_records) == 3
    assert res_no_hdr.valid_records[0].record_index == 0
    assert res_no_hdr.valid_records[0].source_line_number == 1
    assert res_no_hdr.valid_records[0].raw_text == "text"
    assert res_no_hdr.valid_records[1].raw_text == "မြန်မာစာ နမူနာ ၁"


def test_csv_malformed_column_count_isolation() -> None:
    """Verify CSV rows with too many or too few columns are isolated as MalformedRecord."""
    csv_bytes = (
        "id,text\n"
        "1,Valid row one\n"
        "2,Too,Many,Columns\n"
        "3\n"
        "4,Valid row four\n"
    ).encode("utf-8")

    result = load_dataset(csv_bytes, "ragged.csv", csv_has_header=True)

    assert result.analyzed_raw_records == 4
    assert len(result.valid_records) == 2
    assert len(result.malformed_records) == 2

    # Valid records retain their sequential record_index (0 and 3)
    assert result.valid_records[0].record_index == 0
    assert result.valid_records[0].source_line_number == 2
    assert result.valid_records[1].record_index == 3
    assert result.valid_records[1].source_line_number == 5

    # Malformed records captured at record_index 1 (line 3) and record_index 2 (line 4)
    assert result.malformed_records[0].record_index == 1
    assert result.malformed_records[0].source_line_number == 3
    assert "column count mismatch" in result.malformed_records[0].reason.lower()
    assert result.malformed_records[1].record_index == 2
    assert result.malformed_records[1].source_line_number == 4


def test_jsonl_valid_missing_empty_and_malformed_lines() -> None:
    """Verify JSONL loader distinguishes valid strings, empty strings, null/missing fields, and broken lines."""
    jsonl_content = "\n".join(
        [
            '{"id": 1, "text": "မင်္ဂလာပါ"}',          # Line 1: Valid non-empty
            '{"id": 2, "text": ""}',                    # Line 2: Empty string
            '{"id": 3, "text": null}',                  # Line 3: Null value (missing)
            '{"id": 4, "other": "no text key"}',        # Line 4: Missing key (missing)
            "",                                         # Line 5: Blank line -> MalformedRecord
            '{"id": 6, "text": unquoted_syntax_error}', # Line 6: Invalid JSON -> MalformedRecord
            '["array", "not", "object"]',               # Line 7: Non-dict JSON -> MalformedRecord
            '{"id": 8, "text": 12345}',                 # Line 8: Non-string field -> MalformedRecord
        ]
    ).encode("utf-8")

    result = load_dataset(jsonl_content, "mixed.jsonl", selected_field="text")

    assert result.analyzed_raw_records == 8
    assert len(result.valid_records) == 4
    assert len(result.malformed_records) == 4

    # Check valid records (indices 0, 1, 2, 3)
    assert result.valid_records[0].raw_text == "မင်္ဂလာပါ"
    assert result.valid_records[0].is_missing_value is False

    assert result.valid_records[1].raw_text == ""
    assert result.valid_records[1].is_missing_value is False

    assert result.valid_records[2].raw_text is None
    assert result.valid_records[2].is_missing_value is True

    assert result.valid_records[3].raw_text is None
    assert result.valid_records[3].is_missing_value is True

    # Check malformed records (indices 4, 5, 6, 7 -> lines 5, 6, 7, 8)
    assert [m.source_line_number for m in result.malformed_records] == [5, 6, 7, 8]
    assert "Blank line" in result.malformed_records[0].reason
    assert "Invalid JSON syntax" in result.malformed_records[1].reason
    assert "not a JSON object" in result.malformed_records[2].reason
    assert "Non-string" in result.malformed_records[3].reason


def test_encoding_error_line_isolation_in_txt_and_jsonl() -> None:
    """Verify invalid UTF-8 bytes on a single line do not crash or discard valid lines."""
    raw_bytes = (
        b"Valid line 1\n"
        b"Corrupted byte \xff\xfe here\n"
        b"Valid line 3\n"
    )
    result = load_dataset(raw_bytes, "partial_bad_utf8.txt")

    assert result.analyzed_raw_records == 3
    assert len(result.valid_records) == 2
    assert len(result.malformed_records) == 1
    assert result.malformed_records[0].source_line_number == 2
    assert "UTF-8 decoding error" in result.malformed_records[0].reason


def test_unsupported_extension_and_binary_rejection() -> None:
    """Verify unsupported file extensions and binary NUL streams raise explicit errors."""
    with pytest.raises(UnsupportedFormatError):
        load_dataset(b"some content", "spreadsheet.xlsx")

    # UTF-16 BOM raises FileEncodingError
    with pytest.raises(FileEncodingError):
        load_dataset(b"\xff\xfeH\x00e\x00l\x00l\x00o\x00", "utf16.txt")

    # Binary NUL bytes raise UnsupportedFormatError
    with pytest.raises(UnsupportedFormatError):
        load_dataset(b"PK\x03\x04\x00\x00binary_zip_stream", "fake.csv")


def test_empty_file_handling() -> None:
    """Verify 0-byte files and BOM-only files return 0 records cleanly with a warning."""
    empty_res = load_dataset(b"", "empty.csv")
    assert empty_res.analyzed_raw_records == 0
    assert len(empty_res.valid_records) == 0
    assert len(empty_res.load_warnings) == 1

    bom_only_res = load_dataset(b"\xef\xbb\xbf", "bom_only.txt")
    assert bom_only_res.has_bom is True
    assert bom_only_res.analyzed_raw_records == 0


def test_header_only_csv_warning() -> None:
    """Verify CSV with a header row but 0 data rows returns 0 records and a clear warning."""
    res = load_dataset(b"id,text,label\n", "header_only.csv", csv_has_header=True)
    assert res.analyzed_raw_records == 0
    assert any("header row but 0 data rows" in w for w in res.load_warnings)


def test_hard_record_limit_and_truncation_metadata_txt() -> None:
    """Verify TXT loader stops at max_records_limit and sets truncated_at_limit=True."""
    lines = "\n".join([f"Record number {i}" for i in range(50)]).encode("utf-8")
    config = AnalysisConfig(max_records_limit=15)

    result = load_dataset(lines, "fifty.txt", config=config)

    assert result.truncated_at_limit is True
    assert result.analysis_scope == AnalysisScope.TRUNCATED_SUBSET
    assert result.max_records_limit_applied == 15
    assert result.analyzed_raw_records == 15
    assert len(result.valid_records) == 15
    assert result.valid_records[-1].record_index == 14
    assert any("exceeds the configured maximum limit" in w for w in result.load_warnings)


def test_hard_record_limit_exact_boundary_not_truncated() -> None:
    """Verify a dataset with exactly max_records_limit records is marked COMPLETE, not truncated."""
    lines = "\n".join([f"Record {i}" for i in range(10)]).encode("utf-8")
    config = AnalysisConfig(max_records_limit=10)

    result = load_dataset(lines, "ten.txt", config=config)

    assert result.analyzed_raw_records == 10
    assert result.truncated_at_limit is False
    assert result.analysis_scope == AnalysisScope.COMPLETE


def test_hard_record_limit_csv_and_jsonl() -> None:
    """Verify CSV and JSONL loaders also enforce max_records_limit accurately."""
    csv_lines = ["id,text"] + [f"{i},Row {i}" for i in range(30)]
    csv_bytes = "\n".join(csv_lines).encode("utf-8")
    config = AnalysisConfig(max_records_limit=10)

    csv_res = load_dataset(csv_bytes, "thirty.csv", config=config, csv_has_header=True)
    assert csv_res.truncated_at_limit is True
    assert csv_res.analysis_scope == AnalysisScope.TRUNCATED_SUBSET
    assert csv_res.analyzed_raw_records == 10

    jsonl_bytes = "\n".join([f'{{"text": "Item {i}"}}' for i in range(30)]).encode("utf-8")
    jsonl_res = load_dataset(jsonl_bytes, "thirty.jsonl", config=config)
    assert jsonl_res.truncated_at_limit is True
    assert jsonl_res.analysis_scope == AnalysisScope.TRUNCATED_SUBSET
    assert jsonl_res.analyzed_raw_records == 10


def test_csv_delimiter_autodetection_tsv_and_semicolon() -> None:
    """Verify tab-separated and semicolon-separated CSVs are detected and parsed accurately."""
    tsv_bytes = "id\ttext\n1\tမင်္ဂလာပါ\n2\tကျေးဇူးတင်ပါတယ်\n".encode("utf-8")
    res = load_dataset(tsv_bytes, "tabbed.csv")
    assert res.detected_delimiter == "\t"
    assert len(res.valid_records) == 2
    assert res.valid_records[0].raw_text == "မင်္ဂလာပါ"