"""Dataset parsing, schema inspection, and malformed-record isolation for KhaingLens."""

import csv
import io
import json
from pathlib import Path
from typing import Any

from khainglens.models import (
    AnalysisConfig,
    AnalysisScope,
    FileEncodingError,
    InputFormat,
    LoadResult,
    MalformedRecord,
    ParsedRecord,
    SchemaInspectionResult,
    UnsupportedFormatError,
    build_excerpt,
)

UTF8_BOM = b"\xef\xbb\xbf"
UTF16_LE_BOM = b"\xff\xfe"
UTF16_BE_BOM = b"\xfe\xff"

PREFERRED_TEXT_FIELD_NAMES = (
    "text",
    "content",
    "sentence",
    "burmese",
    "myanmar",
    "body",
    "input",
    "prompt",
    "message",
    "document",
)


def detect_input_format(
    filename: str, explicit_format: InputFormat | None = None
) -> InputFormat:
    """Determine InputFormat from explicit override or file extension."""
    if explicit_format is not None:
        return explicit_format

    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        return InputFormat.CSV
    if ext in (".jsonl", ".ndjson"):
        return InputFormat.JSONL
    if ext == ".txt":
        return InputFormat.TXT

    raise UnsupportedFormatError(
        f"Unsupported file format '{ext or 'no extension'}' for file '{filename}'. "
        "KhaingLens v0.1.0 supports .csv, .jsonl, and .txt files."
    )


def _strip_bom_and_validate_binary(file_bytes: bytes, filename: str) -> tuple[bytes, bool, str]:
    """Check for UTF-8 BOM, reject UTF-16 or binary files, and return (clean_bytes, has_bom, encoding_label)."""
    if file_bytes.startswith(UTF16_LE_BOM) or file_bytes.startswith(UTF16_BE_BOM):
        raise FileEncodingError(
            f"File '{filename}' appears to be encoded in UTF-16 (detected byte-order mark). "
            "Please convert the file to UTF-8 before inspecting in KhaingLens."
        )

    has_bom = False
    clean_bytes = file_bytes
    if file_bytes.startswith(UTF8_BOM):
        has_bom = True
        clean_bytes = file_bytes[len(UTF8_BOM) :]

    # Check the first 8 KB for NUL bytes which indicate binary files (e.g. XLSX/PDF/PNG/UTF-16)
    sample = clean_bytes[:8192]
    if b"\x00" in sample:
        raise UnsupportedFormatError(
            f"File '{filename}' contains binary NUL bytes (0x00) and cannot be parsed as UTF-8 text."
        )

    encoding_label = "utf-8-sig" if has_bom else "utf-8"
    return clean_bytes, has_bom, encoding_label


def _choose_suggested_field(candidate_fields: list[str]) -> str | None:
    """Pick the most likely text field from candidate column/key names."""
    if not candidate_fields:
        return None
    lower_map = {field.lower().strip(): field for field in candidate_fields}
    for pref in PREFERRED_TEXT_FIELD_NAMES:
        if pref in lower_map:
            return lower_map[pref]
    return candidate_fields[0]


def _detect_csv_delimiter(sample_text: str, explicit_delimiter: str | None = None) -> str:
    """Detect CSV delimiter from a text sample using csv.Sniffer with safe fallback to ','."""
    if explicit_delimiter is not None and len(explicit_delimiter) == 1:
        return explicit_delimiter
    if not sample_text.strip():
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
        return dialect.delimiter
    except csv.Error:
        return ","


def inspect_file_schema(
    file_bytes: bytes,
    filename: str,
    explicit_format: InputFormat | None = None,
    csv_delimiter: str | None = None,
    csv_has_header: bool = True,
) -> SchemaInspectionResult:
    """Inspect candidate fields/columns, encoding, and preview rows without loading the full dataset."""
    try:
        fmt = detect_input_format(filename, explicit_format)
        clean_bytes, has_bom, encoding_label = _strip_bom_and_validate_binary(file_bytes, filename)
    except (UnsupportedFormatError, FileEncodingError) as exc:
        return SchemaInspectionResult(
            file_format=explicit_format or InputFormat.TXT,
            encoding_detected="unknown",
            has_bom=False,
            csv_has_header=csv_has_header,
            error_message=str(exc),
        )

    if len(clean_bytes) == 0:
        return SchemaInspectionResult(
            file_format=fmt,
            encoding_detected=encoding_label,
            has_bom=has_bom,
            csv_has_header=csv_has_header,
            candidate_fields=[],
            suggested_field=None,
            detected_delimiter="," if fmt == InputFormat.CSV else None,
            preview_rows=[],
        )

    sample_bytes = clean_bytes[:65536]
    sample_text = sample_bytes.decode("utf-8", errors="replace")

    if fmt == InputFormat.TXT:
        lines = sample_text.splitlines()[:5]
        preview = [{"line": idx + 1, "text": line[:200]} for idx, line in enumerate(lines)]
        return SchemaInspectionResult(
            file_format=InputFormat.TXT,
            encoding_detected=encoding_label,
            has_bom=has_bom,
            csv_has_header=False,
            candidate_fields=["text"],
            suggested_field="text",
            detected_delimiter=None,
            preview_rows=preview,
        )

    if fmt == InputFormat.CSV:
        delim = _detect_csv_delimiter(sample_text, csv_delimiter)
        reader = csv.reader(io.StringIO(sample_text), delimiter=delim)
        rows: list[list[str]] = []
        try:
            for _ in range(6):
                row = next(reader, None)
                if row is None:
                    break
                rows.append(row)
        except csv.Error as exc:
            return SchemaInspectionResult(
                file_format=InputFormat.CSV,
                encoding_detected=encoding_label,
                has_bom=has_bom,
                csv_has_header=csv_has_header,
                detected_delimiter=delim,
                error_message=f"CSV header/preview parse error: {exc}",
            )

        if not rows:
            return SchemaInspectionResult(
                file_format=InputFormat.CSV,
                encoding_detected=encoding_label,
                has_bom=has_bom,
                csv_has_header=csv_has_header,
                detected_delimiter=delim,
            )

        if csv_has_header:
            headers = [
                h if h != "" else f"unnamed_col_{i}" for i, h in enumerate(rows[0])
            ]
            data_rows = rows[1:6]
        else:
            headers = [f"col_{i}" for i in range(len(rows[0]))]
            data_rows = rows[:5]

        preview_rows = []
        for r in data_rows:
            preview_rows.append(
                {
                    headers[i]: (r[i][:200] if i < len(r) else None)
                    for i in range(len(headers))
                }
            )

        return SchemaInspectionResult(
            file_format=InputFormat.CSV,
            encoding_detected=encoding_label,
            has_bom=has_bom,
            csv_has_header=csv_has_header,
            candidate_fields=headers,
            suggested_field=_choose_suggested_field(headers),
            detected_delimiter=delim,
            preview_rows=preview_rows,
        )

    # JSONL schema inspection
    candidate_keys: list[str] = []
    seen_keys: set[str] = set()
    preview_rows = []

    for raw_line in sample_text.splitlines():
        if not raw_line.strip():
            continue
        try:
            parsed = json.loads(raw_line)
            if isinstance(parsed, dict):
                for k in parsed.keys():
                    if k not in seen_keys:
                        seen_keys.add(k)
                        candidate_keys.append(str(k))
                if len(preview_rows) < 5:
                    preview_rows.append(
                        {str(k): str(v)[:200] if v is not None else None for k, v in parsed.items()}
                    )
        except json.JSONDecodeError:
            continue

    return SchemaInspectionResult(
        file_format=InputFormat.JSONL,
        encoding_detected=encoding_label,
        has_bom=has_bom,
        csv_has_header=False,
        candidate_fields=candidate_keys,
        suggested_field=_choose_suggested_field(candidate_keys),
        detected_delimiter=None,
        preview_rows=preview_rows,
    )


def load_dataset(
    file_bytes: bytes,
    filename: str,
    config: AnalysisConfig | None = None,
    explicit_format: InputFormat | None = None,
    selected_field: str | None = None,
    csv_delimiter: str | None = None,
    csv_has_header: bool = True,
) -> LoadResult:
    """Parse a CSV, JSONL, or TXT dataset with strict malformed-record isolation and record limits."""
    active_config = config or AnalysisConfig()
    fmt = detect_input_format(filename, explicit_format)
    clean_bytes, has_bom, encoding_label = _strip_bom_and_validate_binary(file_bytes, filename)

    if len(clean_bytes) == 0:
        return LoadResult(
            filename=filename,
            file_format=fmt,
            encoding_used=encoding_label,
            has_bom=has_bom,
            csv_has_header=csv_has_header if fmt == InputFormat.CSV else None,
            selected_field=selected_field,
            detected_delimiter=csv_delimiter if fmt == InputFormat.CSV else None,
            analysis_scope=AnalysisScope.COMPLETE,
            truncated_at_limit=False,
            max_records_limit_applied=active_config.max_records_limit,
            analyzed_raw_records=0,
            valid_records=[],
            malformed_records=[],
            load_warnings=["Uploaded file is empty (0 bytes after header inspection)."],
        )

    if fmt == InputFormat.TXT:
        return _load_txt_stream(
            clean_bytes=clean_bytes,
            filename=filename,
            encoding_label=encoding_label,
            has_bom=has_bom,
            config=active_config,
        )
    if fmt == InputFormat.JSONL:
        return _load_jsonl_stream(
            clean_bytes=clean_bytes,
            filename=filename,
            encoding_label=encoding_label,
            has_bom=has_bom,
            selected_field=selected_field,
            config=active_config,
        )
    return _load_csv_stream(
        clean_bytes=clean_bytes,
        filename=filename,
        encoding_label=encoding_label,
        has_bom=has_bom,
        selected_field=selected_field,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
        config=active_config,
    )


def _load_txt_stream(
    clean_bytes: bytes,
    filename: str,
    encoding_label: str,
    has_bom: bool,
    config: AnalysisConfig,
) -> LoadResult:
    """Load TXT dataset line-by-line where each physical line is one record."""
    valid_records: list[ParsedRecord] = []
    malformed_records: list[MalformedRecord] = []
    warnings: list[str] = []
    truncated = False

    stream = io.BytesIO(clean_bytes)
    record_idx = 0
    line_num = 0

    for raw_line_bytes in stream:
        if record_idx >= config.max_records_limit:
            truncated = True
            break

        line_num += 1
        # Strip trailing line ending (\r\n or \n) while preserving all internal/leading/trailing spaces
        if raw_line_bytes.endswith(b"\r\n"):
            content_bytes = raw_line_bytes[:-2]
        elif raw_line_bytes.endswith(b"\n") or raw_line_bytes.endswith(b"\r"):
            content_bytes = raw_line_bytes[:-1]
        else:
            content_bytes = raw_line_bytes

        try:
            decoded_text = content_bytes.decode("utf-8", errors="strict")
            valid_records.append(
                ParsedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_text=decoded_text,
                    is_missing_value=False,
                )
            )
        except UnicodeDecodeError as exc:
            safe_preview = content_bytes.decode("utf-8", errors="backslashreplace")
            malformed_records.append(
                MalformedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_excerpt=build_excerpt(safe_preview, config.excerpt_max_chars),
                    reason="UTF-8 decoding error",
                    error_detail=f"Invalid UTF-8 byte sequence at byte offset {exc.start}: {exc.reason}",
                )
            )
        record_idx += 1

    scope = AnalysisScope.TRUNCATED_SUBSET if truncated else AnalysisScope.COMPLETE
    if truncated:
        warnings.append(
            f"Dataset exceeds the configured maximum limit of {config.max_records_limit:,} records. "
            f"Only the first {config.max_records_limit:,} records were analyzed."
        )

    return LoadResult(
        filename=filename,
        file_format=InputFormat.TXT,
        encoding_used=encoding_label,
        has_bom=has_bom,
        csv_has_header=None,
        selected_field="text",
        detected_delimiter=None,
        analysis_scope=scope,
        truncated_at_limit=truncated,
        max_records_limit_applied=config.max_records_limit,
        analyzed_raw_records=len(valid_records) + len(malformed_records),
        valid_records=valid_records,
        malformed_records=malformed_records,
        load_warnings=warnings,
    )


def _load_jsonl_stream(
    clean_bytes: bytes,
    filename: str,
    encoding_label: str,
    has_bom: bool,
    selected_field: str | None,
    config: AnalysisConfig,
) -> LoadResult:
    """Load JSONL dataset line-by-line, isolating blank lines, broken JSON, and non-string fields."""
    valid_records: list[ParsedRecord] = []
    malformed_records: list[MalformedRecord] = []
    warnings: list[str] = []
    truncated = False

    # Determine target field if not explicitly provided
    target_field = selected_field
    if not target_field:
        schema = inspect_file_schema(clean_bytes, filename, explicit_format=InputFormat.JSONL)
        target_field = schema.suggested_field

    stream = io.BytesIO(clean_bytes)
    record_idx = 0
    line_num = 0

    for raw_line_bytes in stream:
        if record_idx >= config.max_records_limit:
            truncated = True
            break

        line_num += 1
        stripped_bytes = raw_line_bytes.strip()

        # Policy: Blank lines inside a JSONL file are reported as MalformedRecord
        if stripped_bytes == b"":
            malformed_records.append(
                MalformedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_excerpt="",
                    reason="Blank line in JSONL file",
                    error_detail=f"Line {line_num} is empty or whitespace-only; expected a JSON object.",
                )
            )
            record_idx += 1
            continue

        try:
            decoded_line = raw_line_bytes.decode("utf-8", errors="strict").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            safe_preview = raw_line_bytes.decode("utf-8", errors="backslashreplace").rstrip("\r\n")
            malformed_records.append(
                MalformedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_excerpt=build_excerpt(safe_preview, config.excerpt_max_chars),
                    reason="UTF-8 decoding error",
                    error_detail=f"Invalid UTF-8 byte sequence at byte offset {exc.start}: {exc.reason}",
                )
            )
            record_idx += 1
            continue

        try:
            obj = json.loads(decoded_line)
        except json.JSONDecodeError as exc:
            malformed_records.append(
                MalformedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_excerpt=build_excerpt(decoded_line, config.excerpt_max_chars),
                    reason="Invalid JSON syntax",
                    error_detail=f"{exc.msg} (column {exc.colno})",
                )
            )
            record_idx += 1
            continue

        if not isinstance(obj, dict):
            malformed_records.append(
                MalformedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_excerpt=build_excerpt(decoded_line, config.excerpt_max_chars),
                    reason="JSONL record is not a JSON object",
                    error_detail=f"Expected JSON object (dict), got {type(obj).__name__}.",
                )
            )
            record_idx += 1
            continue

        # If target_field was still None (e.g. first valid line was beyond 64KB), pick from first valid object
        if target_field is None and obj:
            target_field = _choose_suggested_field([str(k) for k in obj.keys()])

        if target_field is None or target_field not in obj or obj[target_field] is None:
            extra_preview = {
                str(k): str(v)[:100]
                for k, v in list(obj.items())[:5]
                if k != target_field
            }
            valid_records.append(
                ParsedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_text=None,
                    is_missing_value=True,
                    extra_fields_preview=extra_preview,
                )
            )
            record_idx += 1
            continue

        val = obj[target_field]
        if not isinstance(val, str):
            malformed_records.append(
                MalformedRecord(
                    record_index=record_idx,
                    source_line_number=line_num,
                    raw_excerpt=build_excerpt(decoded_line, config.excerpt_max_chars),
                    reason="Non-string text field value",
                    error_detail=(
                        f"Field '{target_field}' contains {type(val).__name__} "
                        f"({repr(val)[:80]}) instead of string."
                    ),
                )
            )
            record_idx += 1
            continue

        extra_preview = {
            str(k): str(v)[:100]
            for k, v in list(obj.items())[:5]
            if k != target_field
        }
        valid_records.append(
            ParsedRecord(
                record_index=record_idx,
                source_line_number=line_num,
                raw_text=val,
                is_missing_value=False,
                extra_fields_preview=extra_preview,
            )
        )
        record_idx += 1

    scope = AnalysisScope.TRUNCATED_SUBSET if truncated else AnalysisScope.COMPLETE
    if truncated:
        warnings.append(
            f"Dataset exceeds the configured maximum limit of {config.max_records_limit:,} records. "
            f"Only the first {config.max_records_limit:,} records were analyzed."
        )

    return LoadResult(
        filename=filename,
        file_format=InputFormat.JSONL,
        encoding_used=encoding_label,
        has_bom=has_bom,
        csv_has_header=None,
        selected_field=target_field,
        detected_delimiter=None,
        analysis_scope=scope,
        truncated_at_limit=truncated,
        max_records_limit_applied=config.max_records_limit,
        analyzed_raw_records=len(valid_records) + len(malformed_records),
        valid_records=valid_records,
        malformed_records=malformed_records,
        load_warnings=warnings,
    )


def _iter_decoded_csv_lines(
    clean_bytes: bytes,
    malformed_records_out: list[MalformedRecord],
    excerpt_max_chars: int,
    state: dict[str, int],
):
    """Yield valid UTF-8 lines to csv.reader while recording undecodable lines in malformed_records_out."""
    stream = io.BytesIO(clean_bytes)
    for raw_line_bytes in stream:
        state["physical_line"] += 1
        try:
            decoded = raw_line_bytes.decode("utf-8", errors="strict")
            yield decoded
        except UnicodeDecodeError as exc:
            if state["record_idx"] < state["max_limit"]:
                safe_preview = raw_line_bytes.decode("utf-8", errors="backslashreplace").rstrip(
                    "\r\n"
                )
                malformed_records_out.append(
                    MalformedRecord(
                        record_index=state["record_idx"],
                        source_line_number=state["physical_line"],
                        raw_excerpt=build_excerpt(safe_preview, excerpt_max_chars),
                        reason="UTF-8 decoding error in CSV row",
                        error_detail=(
                            f"Invalid UTF-8 byte sequence at line {state['physical_line']}, "
                            f"byte offset {exc.start}: {exc.reason}"
                        ),
                    )
                )
                state["record_idx"] += 1
            else:
                state["truncated"] = True


def _load_csv_stream(
    clean_bytes: bytes,
    filename: str,
    encoding_label: str,
    has_bom: bool,
    selected_field: str | None,
    csv_delimiter: str | None,
    csv_has_header: bool,
    config: AnalysisConfig,
) -> LoadResult:
    """Load CSV dataset with header/headerless toggle, column mismatch detection, and record limits."""
    valid_records: list[ParsedRecord] = []
    malformed_records: list[MalformedRecord] = []
    warnings: list[str] = []

    sample_text = clean_bytes[:65536].decode("utf-8", errors="replace")
    delim = _detect_csv_delimiter(sample_text, csv_delimiter)

    state: dict[str, int] = {
        "physical_line": 0,
        "record_idx": 0,
        "max_limit": config.max_records_limit,
        "truncated": 0,
    }

    line_iter = _iter_decoded_csv_lines(
        clean_bytes=clean_bytes,
        malformed_records_out=malformed_records,
        excerpt_max_chars=config.excerpt_max_chars,
        state=state,
    )
    reader = csv.reader(line_iter, delimiter=delim)

    headers: list[str] = []
    target_col_idx: int = 0
    target_field_name: str | None = selected_field

    try:
        first_row = next(reader, None)
    except csv.Error as exc:
        malformed_records.append(
            MalformedRecord(
                record_index=0,
                source_line_number=max(1, state["physical_line"]),
                raw_excerpt=build_excerpt(sample_text.splitlines()[0] if sample_text else "", config.excerpt_max_chars),
                reason="Malformed CSV syntax on first row",
                error_detail=str(exc),
            )
        )
        return LoadResult(
            filename=filename,
            file_format=InputFormat.CSV,
            encoding_used=encoding_label,
            has_bom=has_bom,
            csv_has_header=csv_has_header,
            selected_field=selected_field,
            detected_delimiter=delim,
            analysis_scope=AnalysisScope.COMPLETE,
            truncated_at_limit=False,
            max_records_limit_applied=config.max_records_limit,
            analyzed_raw_records=len(malformed_records),
            valid_records=[],
            malformed_records=malformed_records,
            load_warnings=warnings,
        )

    if first_row is None:
        return LoadResult(
            filename=filename,
            file_format=InputFormat.CSV,
            encoding_used=encoding_label,
            has_bom=has_bom,
            csv_has_header=csv_has_header,
            selected_field=selected_field,
            detected_delimiter=delim,
            analysis_scope=AnalysisScope.COMPLETE,
            truncated_at_limit=False,
            max_records_limit_applied=config.max_records_limit,
            analyzed_raw_records=len(malformed_records),
            valid_records=[],
            malformed_records=malformed_records,
            load_warnings=["CSV file contains no parseable rows."],
        )

    first_row_line_num = state["physical_line"]

    if csv_has_header:
        headers = [
            col if col != "" else f"unnamed_col_{idx}"
            for idx, col in enumerate(first_row)
        ]
    else:
        headers = [f"col_{idx}" for idx in range(len(first_row))]

    if target_field_name and target_field_name in headers:
        target_col_idx = headers.index(target_field_name)
    else:
        suggested = _choose_suggested_field(headers)
        target_field_name = suggested
        target_col_idx = headers.index(suggested) if suggested in headers else 0

    expected_num_cols = len(headers)

    def _process_csv_data_row(row: list[str], row_line_num: int) -> None:
        if state["record_idx"] >= config.max_records_limit:
            state["truncated"] = 1
            return

        rec_idx = state["record_idx"]
        state["record_idx"] += 1

        if len(row) != expected_num_cols:
            row_preview = delim.join(row)
            malformed_records.append(
                MalformedRecord(
                    record_index=rec_idx,
                    source_line_number=row_line_num,
                    raw_excerpt=build_excerpt(row_preview, config.excerpt_max_chars),
                    reason="CSV column count mismatch",
                    error_detail=(
                        f"Expected {expected_num_cols} columns ({headers}), "
                        f"but row has {len(row)} column(s)."
                    ),
                )
            )
            return

        cell_value = row[target_col_idx]
        extra_preview = {
            headers[i]: row[i][:100]
            for i in range(min(len(headers), 6))
            if i != target_col_idx
        }
        valid_records.append(
            ParsedRecord(
                record_index=rec_idx,
                source_line_number=row_line_num,
                raw_text=cell_value,
                is_missing_value=False,
                extra_fields_preview=extra_preview,
            )
        )

    # If csv_has_header is False, Row 1 is a data record!
    if not csv_has_header:
        _process_csv_data_row(first_row, first_row_line_num)

    while True:
        try:
            row = next(reader, None)
            if row is None:
                break
            if state["record_idx"] >= config.max_records_limit:
                state["truncated"] = 1
                break
            _process_csv_data_row(row, state["physical_line"])
        except csv.Error as exc:
            if state["record_idx"] >= config.max_records_limit:
                state["truncated"] = 1
                break
            rec_idx = state["record_idx"]
            state["record_idx"] += 1
            malformed_records.append(
                MalformedRecord(
                    record_index=rec_idx,
                    source_line_number=max(1, state["physical_line"]),
                    raw_excerpt="<Unclosed quote or malformed CSV syntax>",
                    reason="CSV syntax error",
                    error_detail=f"CSV parser error near line {state['physical_line']}: {exc}",
                )
            )
            break

    if csv_has_header and len(valid_records) == 0 and len(malformed_records) == 0:
        warnings.append("CSV file contains a header row but 0 data rows.")

    # Ensure records are ordered by record_index
    valid_records.sort(key=lambda r: r.record_index)
    malformed_records.sort(key=lambda r: r.record_index)

    truncated = bool(state["truncated"])
    scope = AnalysisScope.TRUNCATED_SUBSET if truncated else AnalysisScope.COMPLETE
    if truncated:
        warnings.append(
            f"Dataset exceeds the configured maximum limit of {config.max_records_limit:,} records. "
            f"Only the first {config.max_records_limit:,} records were analyzed."
        )

    return LoadResult(
        filename=filename,
        file_format=InputFormat.CSV,
        encoding_used=encoding_label,
        has_bom=has_bom,
        csv_has_header=csv_has_header,
        selected_field=target_field_name,
        detected_delimiter=delim,
        analysis_scope=scope,
        truncated_at_limit=truncated,
        max_records_limit_applied=config.max_records_limit,
        analyzed_raw_records=len(valid_records) + len(malformed_records),
        valid_records=valid_records,
        malformed_records=malformed_records,
        load_warnings=warnings,
    )