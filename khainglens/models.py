"""Shared data models, controlled enums, and schemas for KhaingLens."""

from enum import Enum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class InputFormat(str, Enum):
    """Supported dataset file formats in KhaingLens v0.1.0."""

    CSV = "csv"
    JSONL = "jsonl"
    TXT = "txt"


class Severity(str, Enum):
    """Controlled severity levels for dataset anomalies."""

    ERROR = "error"       # Definite parsing, encoding, or structural problem
    WARNING = "warning"   # Potentially problematic record requiring human review
    INFO = "info"         # Neutral observation or statistical pattern


class AnalysisScope(str, Enum):
    """Indicates whether the entire file or a truncated subset was analyzed."""

    COMPLETE = "complete_dataset"
    TRUNCATED_SUBSET = "truncated_subset"


class AnomalyType(str, Enum):
    """Controlled vocabulary of all anomaly categories detected by KhaingLens."""

    # Parsing & Missingness
    MALFORMED_RECORD = "malformed_record"
    MISSING_TEXT_VALUE = "missing_text_value"
    EMPTY_STRING = "empty_string"
    WHITESPACE_ONLY = "whitespace_only"

    # Whitespace Anomalies
    LEADING_WHITESPACE = "leading_whitespace"
    TRAILING_WHITESPACE = "trailing_whitespace"
    REPEATED_SPACES = "repeated_spaces"
    UNUSUAL_TAB_OR_NEWLINE = "unusual_tab_or_newline"

    # Duplicates
    EXACT_DUPLICATE = "exact_duplicate"
    NORMALIZED_DUPLICATE = "normalized_duplicate"

    # Unicode & Burmese Diagnostics
    UNEXPECTED_CONTROL_CHAR = "unexpected_control_char"
    EMBEDDED_BOM = "embedded_bom"
    ZERO_WIDTH_CHARACTER = "zero_width_character"
    UNUSUAL_UNICODE_SPACE = "unusual_unicode_space"
    UNICODE_NORMALIZATION_DIFF = "unicode_normalization_diff"
    MYANMAR_ORPHAN_COMBINING = "myanmar_orphan_combining"
    MYANMAR_DUPLICATE_DIACRITIC = "myanmar_duplicate_diacritic"
    MYANMAR_SUSPECTED_ZAWGYI = "myanmar_suspected_zawgyi"

    # Length Outliers
    LENGTH_OUTLIER_SHORT = "length_outlier_short"
    LENGTH_OUTLIER_LONG = "length_outlier_long"


class KhaingLensError(Exception):
    """Base exception for explicit, user-facing KhaingLens errors."""


class UnsupportedFormatError(KhaingLensError):
    """Raised when an uploaded file has an unsupported format or binary signature."""


class FileEncodingError(KhaingLensError):
    """Raised when a file cannot be decoded using supported UTF-8 encodings."""


class AnalysisConfig(BaseModel):
    """User-configurable thresholds and parameters for dataset inspection."""

    model_config = ConfigDict(frozen=True)

    iqr_multiplier: float = Field(
        default=1.5,
        ge=0.5,
        le=5.0,
        description="Multiplier (k) for Interquartile Range (IQR) outlier fences.",
    )
    min_records_for_iqr: int = Field(
        default=4,
        ge=4,
        description="Minimum non-empty analyzable records required for IQR calculation.",
    )
    repeated_space_threshold: int = Field(
        default=2,
        ge=2,
        le=20,
        description="Minimum consecutive ASCII spaces to flag as REPEATED_SPACES.",
    )
    detect_normalized_duplicates: bool = Field(
        default=True,
        description="Flag records that differ in raw code points but match under Unicode NFC.",
    )
    flag_internal_newlines: bool = Field(
        default=True,
        description="Flag embedded newline characters inside individual records.",
    )
    flag_tabs: bool = Field(
        default=True,
        description="Flag tab characters inside individual records.",
    )
    max_records_limit: int = Field(
        default=100_000,
        ge=1,
        le=500_000,
        description="Maximum number of records analyzed before truncating with a warning.",
    )
    excerpt_max_chars: int = Field(
        default=300,
        ge=50,
        le=2000,
        description="Maximum character length for text excerpts stored in anomalies.",
    )


class ParsedRecord(BaseModel):
    """An immutable, successfully parsed record from the input dataset."""

    model_config = ConfigDict(frozen=True)

    record_index: int = Field(ge=0)
    source_line_number: int = Field(ge=1)
    raw_text: str | None = None
    is_missing_value: bool = False
    extra_fields_preview: dict[str, str] = Field(default_factory=dict)


class MalformedRecord(BaseModel):
    """An immutable record representing a row or line that failed parsing."""

    model_config = ConfigDict(frozen=True)

    record_index: int = Field(ge=0)
    source_line_number: int = Field(ge=1)
    raw_excerpt: str
    reason: str
    error_detail: str


class SchemaInspectionResult(BaseModel):
    """Lightweight pre-load schema inspection result for UI column/delimiter selection."""

    model_config = ConfigDict(frozen=True)

    file_format: InputFormat
    encoding_detected: str
    has_bom: bool
    csv_has_header: bool = True
    candidate_fields: list[str] = Field(default_factory=list)
    suggested_field: str | None = None
    detected_delimiter: str | None = None
    preview_rows: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


class LoadResult(BaseModel):
    """Complete output of the dataset loader, preserving both valid and malformed records."""

    model_config = ConfigDict(frozen=True)

    filename: str
    file_format: InputFormat
    encoding_used: str
    has_bom: bool
    csv_has_header: bool | None = None
    selected_field: str | None = None
    detected_delimiter: str | None = None
    analysis_scope: AnalysisScope = AnalysisScope.COMPLETE
    truncated_at_limit: bool = False
    max_records_limit_applied: int = 100_000
    analyzed_raw_records: int = Field(ge=0)
    valid_records: list[ParsedRecord] = Field(default_factory=list)
    malformed_records: list[MalformedRecord] = Field(default_factory=list)
    load_warnings: list[str] = Field(default_factory=list)


class Anomaly(BaseModel):
    """A consolidated finding for a single (record_index, issue_type) pair."""

    model_config = ConfigDict(frozen=True)

    anomaly_id: str
    record_index: int = Field(ge=0)
    source_line_number: int | None = None
    issue_type: AnomalyType
    severity: Severity
    title: str
    explanation: str
    original_excerpt: str
    evidence: str
    suggested_action: str
    occurrence_count: int = Field(default=1, ge=1)
    char_position: int | None = None
    char_positions: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DuplicateGroup(BaseModel):
    """A group of non-empty records sharing identical raw text."""

    model_config = ConfigDict(frozen=True)

    group_id: int = Field(ge=1)
    comparison_basis: str = "raw_exact_match"
    representative_excerpt: str
    record_indices: list[int]
    occurrence_count: int = Field(ge=2)
    surplus_count: int = Field(ge=1)


class HistogramBin(BaseModel):
    """Pre-aggregated histogram bin for text length distribution."""

    model_config = ConfigDict(frozen=True)

    bin_start: float
    bin_end: float
    count: int = Field(ge=0)


class DatasetStats(BaseModel):
    """Comprehensive descriptive statistics and anomaly metrics for an analyzed dataset."""

    model_config = ConfigDict(frozen=True)

    # Scope & Truncation Transparency
    analysis_scope: AnalysisScope
    truncated_at_limit: bool
    max_records_limit_applied: int

    # Record Counts (within analyzed scope)
    analyzed_raw_records: int = Field(ge=0)
    successfully_parsed_records: int = Field(ge=0)
    malformed_record_count: int = Field(ge=0)
    missing_value_record_count: int = Field(ge=0)
    empty_string_record_count: int = Field(ge=0)
    whitespace_only_record_count: int = Field(ge=0)
    total_empty_or_missing_records: int = Field(ge=0)
    non_empty_analyzable_records: int = Field(ge=0)

    # Exact Duplication Metrics (over non-empty analyzable records, raw exact match)
    unique_non_empty_texts: int = Field(ge=0)
    duplicate_group_count: int = Field(ge=0)
    total_records_in_duplicate_groups: int = Field(ge=0)
    surplus_duplicate_occurrences: int = Field(ge=0)
    normalized_duplicate_group_count: int = Field(ge=0)

    # Length Metrics (Unicode code points over non-empty analyzable records)
    length_unit: str = "unicode_code_points"
    min_length: int | None = None
    max_length: int | None = None
    mean_length: float | None = None
    median_length: float | None = None
    q1_length: float | None = None
    q3_length: float | None = None
    iqr_length: float | None = None
    lower_outlier_fence: float | None = None
    upper_outlier_fence: float | None = None
    length_histogram: list[HistogramBin] = Field(default_factory=list)

    # Approximate Token Metrics (explicitly labeled as whitespace-delimited)
    mean_approx_whitespace_tokens: float | None = None
    max_approx_whitespace_tokens: int | None = None

    # Anomaly & Rate Metrics
    total_anomaly_findings: int = Field(ge=0)
    total_in_record_anomaly_occurrences: int = Field(ge=0)
    records_with_any_anomaly: int = Field(ge=0)
    anomaly_counts_by_type: dict[str, int] = Field(default_factory=dict)
    occurrence_sums_by_type: dict[str, int] = Field(default_factory=dict)
    affected_records_by_type: dict[str, int] = Field(default_factory=dict)
    anomaly_counts_by_severity: dict[str, int] = Field(default_factory=dict)
    affected_records_by_severity: dict[str, int] = Field(default_factory=dict)

    # Rates (0.0 to 1.0, None when denominator is 0)
    malformed_record_rate: float | None = None
    empty_or_missing_rate: float | None = None
    duplicate_involvement_rate: float | None = None
    surplus_duplicate_rate: float | None = None
    affected_record_rate: float | None = None


class AnalysisResult(BaseModel):
    """Top-level immutable container returned by the KhaingLens analysis pipeline."""

    model_config = ConfigDict(frozen=True)

    analysis_timestamp_utc: str
    config: AnalysisConfig
    load_result: LoadResult
    stats: DatasetStats
    anomalies: list[Anomaly] = Field(default_factory=list)
    duplicate_groups: list[DuplicateGroup] = Field(default_factory=list)


def build_excerpt(text: str | None, max_chars: int = 300, focus_pos: int | None = None) -> str:
    """Create a safe, bounded excerpt of raw text without modifying the source record.

    If the text exceeds max_chars and focus_pos is provided, centers the window
    around focus_pos so the relevant character anomaly remains visible.
    """
    if text is None:
        return "<MISSING / NULL>"
    if text == "":
        return ""
    if len(text) <= max_chars:
        return text

    if focus_pos is not None and 0 <= focus_pos < len(text):
        half_window = max(10, (max_chars - 6) // 2)
        start = max(0, focus_pos - half_window)
        end = min(len(text), start + (max_chars - 6))
        if end == len(text):
            start = max(0, end - (max_chars - 6))
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return f"{prefix}{text[start:end]}{suffix}"

    return f"{text[: max_chars - 3]}..."