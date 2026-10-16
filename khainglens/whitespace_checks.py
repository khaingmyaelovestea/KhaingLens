"""Empty, missing, whitespace-only, and whitespace anomaly checks for KhaingLens."""

from collections import Counter
import re
import unicodedata

from khainglens.models import (
    AnalysisConfig,
    Anomaly,
    AnomalyType,
    ParsedRecord,
    Severity,
    build_excerpt,
)


def _describe_whitespace_chars(text: str) -> dict[str, int]:
    """Return a human-readable frequency map of whitespace code points in text."""
    counts: Counter[str] = Counter()
    for ch in text:
        cp = f"U+{ord(ch):04X}"
        name = unicodedata.name(ch, "WHITESPACE")
        if ch == "\t":
            name = "CHARACTER TABULATION (TAB)"
        elif ch == "\n":
            name = "LINE FEED (LF)"
        elif ch == "\r":
            name = "CARRIAGE RETURN (CR)"
        counts[f"{cp} {name}"] += 1
    return dict(counts)


def _visualize_whitespace(text: str, max_len: int = 120) -> str:
    """Replace spaces, tabs, and newlines with visible symbols for diagnostic evidence."""
    sample = text[:max_len]
    vis = (
        sample.replace(" ", "␣")
        .replace("\t", "[TAB]")
        .replace("\r", "[CR]")
        .replace("\n", "[LF]")
    )
    if len(text) > max_len:
        vis += "..."
    return vis


def check_record_whitespace(
    record: ParsedRecord,
    config: AnalysisConfig | None = None,
) -> list[Anomaly]:
    """Inspect a single ParsedRecord for missingness, emptiness, or whitespace anomalies.

    Enforces consolidation: at most one Anomaly per (record_index, AnomalyType).
    Does not mutate record.raw_text.
    """
    active_config = config or AnalysisConfig()
    idx = record.record_index
    line_num = record.source_line_number
    raw = record.raw_text

    # 1. Missing value (None / null)
    if record.is_missing_value or raw is None:
        return [
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.MISSING_TEXT_VALUE.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.MISSING_TEXT_VALUE,
                severity=Severity.ERROR,
                title="Missing or Null Text Value",
                explanation=(
                    "The target text field is missing or explicitly null in this record. "
                    "Downstream NLP tokenizers or training pipelines may fail on null inputs."
                ),
                original_excerpt="<MISSING / NULL>",
                evidence="raw_text is None (is_missing_value=True)",
                suggested_action=(
                    "Verify the upstream extraction pipeline or filter out records with null text fields."
                ),
                occurrence_count=1,
                char_position=None,
                char_positions=[],
                metadata={"is_missing_value": True},
            )
        ]

    # 2. Empty string ("")
    if raw == "":
        return [
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.EMPTY_STRING.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.EMPTY_STRING,
                severity=Severity.WARNING,
                title="Empty Text String",
                explanation=(
                    "The record contains an empty string of length 0. Unlike null values, "
                    "an empty string is structurally valid but provides no text content."
                ),
                original_excerpt="",
                evidence="String length is 0 Unicode code points (raw_text == '')",
                suggested_action=(
                    "Inspect whether this blank row/line is an accidental blank line or unpopulated cell."
                ),
                occurrence_count=1,
                char_position=0,
                char_positions=[0],
                metadata={"char_length": 0},
            )
        ]

    # 3. Whitespace-only string (len > 0 but strip() == "")
    if raw.strip() == "":
        char_breakdown = _describe_whitespace_chars(raw)
        breakdown_str = ", ".join(f"{count}x {label}" for label, count in char_breakdown.items())
        return [
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.WHITESPACE_ONLY.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.WHITESPACE_ONLY,
                severity=Severity.WARNING,
                title=f"Whitespace-Only Record ({len(raw)} character(s))",
                explanation=(
                    "The record has a non-zero length but consists exclusively of whitespace "
                    "(spaces, tabs, or newline characters) with no visible content."
                ),
                original_excerpt=_visualize_whitespace(raw, active_config.excerpt_max_chars),
                evidence=f"Contains {len(raw)} whitespace character(s): {breakdown_str}",
                suggested_action=(
                    "Review and remove whitespace-only records if non-empty text examples are required."
                ),
                occurrence_count=len(raw),
                char_position=0,
                char_positions=list(range(len(raw))),
                metadata={
                    "char_length": len(raw),
                    "whitespace_breakdown": char_breakdown,
                    "visualized_whitespace": _visualize_whitespace(raw, 120),
                },
            )
        ]

    # 4. Non-empty record whitespace anomalies
    anomalies: list[Anomaly] = []

    # 4a. Leading whitespace
    stripped_left = raw.lstrip()
    leading_len = len(raw) - len(stripped_left)
    if leading_len > 0:
        leading_slice = raw[:leading_len]
        leading_breakdown = _describe_whitespace_chars(leading_slice)
        breakdown_str = ", ".join(f"{c}x {k}" for k, c in leading_breakdown.items())
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.LEADING_WHITESPACE.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.LEADING_WHITESPACE,
                severity=Severity.INFO,
                title=f"Leading Whitespace ({leading_len} character(s))",
                explanation=(
                    "The record begins with leading whitespace characters. While sometimes intentional "
                    "for indentation, leading spaces often result from unstripped text splitting."
                ),
                original_excerpt=build_excerpt(raw, active_config.excerpt_max_chars, focus_pos=0),
                evidence=(
                    f"{leading_len} leading whitespace character(s) at start of record: {breakdown_str} "
                    f"(Preview: '{_visualize_whitespace(raw[: min(len(raw), leading_len + 20)])}')"
                ),
                suggested_action="Check if leading whitespace should be stripped during preprocessing.",
                occurrence_count=leading_len,
                char_position=0,
                char_positions=list(range(leading_len)),
                metadata={
                    "leading_length": leading_len,
                    "whitespace_breakdown": leading_breakdown,
                },
            )
        )

    # 4b. Trailing whitespace
    stripped_right = raw.rstrip()
    trailing_len = len(raw) - len(stripped_right)
    if trailing_len > 0:
        trailing_start = len(stripped_right)
        trailing_slice = raw[trailing_start:]
        trailing_breakdown = _describe_whitespace_chars(trailing_slice)
        breakdown_str = ", ".join(f"{c}x {k}" for k, c in trailing_breakdown.items())
        preview_start = max(0, trailing_start - 20)
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.TRAILING_WHITESPACE.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.TRAILING_WHITESPACE,
                severity=Severity.INFO,
                title=f"Trailing Whitespace ({trailing_len} character(s))",
                explanation=(
                    "The record ends with trailing whitespace characters, which are invisible in most "
                    "text editors and can prevent exact string matching across duplicates."
                ),
                original_excerpt=build_excerpt(
                    raw, active_config.excerpt_max_chars, focus_pos=trailing_start
                ),
                evidence=(
                    f"{trailing_len} trailing whitespace character(s) starting at index {trailing_start}: "
                    f"{breakdown_str} (Preview: '{_visualize_whitespace(raw[preview_start:])}')"
                ),
                suggested_action="Inspect whether trailing spaces should be trimmed before training or deduplication.",
                occurrence_count=trailing_len,
                char_position=trailing_start,
                char_positions=list(range(trailing_start, len(raw))),
                metadata={
                    "trailing_length": trailing_len,
                    "trailing_start_index": trailing_start,
                    "whitespace_breakdown": trailing_breakdown,
                },
            )
        )

    # 4c. Repeated internal ASCII spaces (excluding leading and trailing whitespace spans)
    # Inspect the interior span [leading_len : len(raw) - trailing_len] so leading/trailing spaces
    # are not double-flagged as repeated internal spaces.
    interior_start = leading_len
    interior_end = len(raw) - trailing_len
    if interior_end > interior_start:
        interior_text = raw[interior_start:interior_end]
        threshold = active_config.repeated_space_threshold
        pattern = re.compile(rf" {{{threshold},}}")
        matches = list(pattern.finditer(interior_text))
        if matches:
            span_positions = [interior_start + m.start() for m in matches]
            span_lengths = [m.end() - m.start() for m in matches]
            first_pos = span_positions[0]
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anom_{idx:06d}_{AnomalyType.REPEATED_SPACES.value}",
                    record_index=idx,
                    source_line_number=line_num,
                    issue_type=AnomalyType.REPEATED_SPACES,
                    severity=Severity.INFO,
                    title=f"Repeated Consecutive Spaces ({len(matches)} span(s))",
                    explanation=(
                        f"The record contains {len(matches)} internal span(s) of {threshold} or more "
                        "consecutive ASCII spaces (U+0020). In English and Burmese corpora, multiple "
                        "consecutive spaces are often OCR or web-scraping artifacts."
                    ),
                    original_excerpt=build_excerpt(
                        raw, active_config.excerpt_max_chars, focus_pos=first_pos
                    ),
                    evidence=(
                        f"Found {len(matches)} repeated-space span(s) at character indices "
                        f"{span_positions[:10]} with lengths {span_lengths[:10]}."
                    ),
                    suggested_action=(
                        "Review whether consecutive spaces represent tabular alignment or accidental double spaces."
                    ),
                    occurrence_count=len(matches),
                    char_position=first_pos,
                    char_positions=span_positions,
                    metadata={
                        "span_count": len(matches),
                        "span_positions": span_positions,
                        "span_lengths": span_lengths,
                        "threshold_applied": threshold,
                    },
                )
            )

    # 4d. Unusual tab or embedded newline characters
    special_positions: list[int] = []
    special_Counter: Counter[str] = Counter()
    for pos, ch in enumerate(raw):
        if ch == "\t" and active_config.flag_tabs:
            special_positions.append(pos)
            special_Counter["U+0009 TAB (\\t)"] += 1
        elif ch in ("\n", "\r") and active_config.flag_internal_newlines:
            special_positions.append(pos)
            label = "U+000A LF (\\n)" if ch == "\n" else "U+000D CR (\\r)"
            special_Counter[label] += 1

    if special_positions:
        first_pos = special_positions[0]
        breakdown_str = ", ".join(f"{c}x {k}" for k, c in special_Counter.items())
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.UNUSUAL_TAB_OR_NEWLINE.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.UNUSUAL_TAB_OR_NEWLINE,
                severity=Severity.WARNING,
                title=f"Embedded Tab or Newline Characters ({len(special_positions)} occurrence(s))",
                explanation=(
                    "The record contains embedded tab (\\t) or line-break (\\r, \\n) characters. "
                    "In single-line text datasets or CSV fields, embedded tabs and newlines can "
                    "indicate unescaped delimiters or multi-line concatenation issues."
                ),
                original_excerpt=build_excerpt(
                    raw, active_config.excerpt_max_chars, focus_pos=first_pos
                ),
                evidence=(
                    f"{len(special_positions)} occurrence(s) at positions {special_positions[:15]}: "
                    f"{breakdown_str}"
                ),
                suggested_action=(
                    "Inspect whether multi-line breaks or tabs inside this record are intentional."
                ),
                occurrence_count=len(special_positions),
                char_position=first_pos,
                char_positions=special_positions,
                metadata={
                    "counts_by_character": dict(special_Counter),
                    "positions": special_positions,
                },
            )
        )

    return anomalies