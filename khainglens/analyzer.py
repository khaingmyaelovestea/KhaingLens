"""End-to-end analysis pipeline orchestration for KhaingLens."""

from datetime import datetime, timezone

from khainglens.duplicate_detection import detect_duplicates
from khainglens.loader import load_dataset
from khainglens.models import (
    AnalysisConfig,
    AnalysisResult,
    Anomaly,
    AnomalyType,
    InputFormat,
    LoadResult,
    MalformedRecord,
    ParsedRecord,
    Severity,
)
from khainglens.outlier_detection import detect_length_outliers
from khainglens.statistics import compute_dataset_statistics
from khainglens.unicode_checks import check_record_unicode
from khainglens.whitespace_checks import check_record_whitespace


def _malformed_record_to_anomaly(malformed: MalformedRecord) -> Anomaly:
    """Convert a MalformedRecord from the loader into a standardized ERROR Anomaly."""
    idx = malformed.record_index
    return Anomaly(
        anomaly_id=f"anom_{idx:06d}_{AnomalyType.MALFORMED_RECORD.value}",
        record_index=idx,
        source_line_number=malformed.source_line_number,
        issue_type=AnomalyType.MALFORMED_RECORD,
        severity=Severity.ERROR,
        title=f"Malformed Record: {malformed.reason}",
        explanation=(
            "This row or line could not be parsed as a valid text record. "
            "KhaingLens isolated the malformed line so the remaining records could be "
            "inspected without silently dropping data."
        ),
        original_excerpt=malformed.raw_excerpt,
        evidence=f"Line {malformed.source_line_number}: {malformed.error_detail}",
        suggested_action=(
            "Inspect the source file at the reported line number to fix syntax, quoting, "
            "column count, or character encoding issues."
        ),
        occurrence_count=1,
        char_position=None,
        char_positions=[],
        metadata={
            "reason": malformed.reason,
            "error_detail": malformed.error_detail,
            "source_line_number": malformed.source_line_number,
        },
    )


def build_record_lookup(
    load_result: LoadResult,
) -> dict[int, ParsedRecord | MalformedRecord]:
    """Build an O(1) dictionary mapping record_index to its ParsedRecord or MalformedRecord."""
    lookup: dict[int, ParsedRecord | MalformedRecord] = {}
    for valid_rec in load_result.valid_records:
        lookup[valid_rec.record_index] = valid_rec
    for mal_rec in load_result.malformed_records:
        lookup[mal_rec.record_index] = mal_rec
    return lookup


def analyze_dataset(
    load_result: LoadResult,
    config: AnalysisConfig | None = None,
) -> AnalysisResult:
    """Run all KhaingLens quality checks and statistical aggregations on a LoadResult.

    Guarantees:
    - Does not mutate load_result or any ParsedRecord / MalformedRecord.
    - Consolidates findings so at most one Anomaly exists per (record_index, issue_type).
    - Accounts for malformed records alongside valid records.
    """
    active_config = config or AnalysisConfig()
    timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    consolidated_map: dict[tuple[int, AnomalyType], Anomaly] = {}

    def _register_anomalies(items: list[Anomaly]) -> None:
        for item in items:
            key = (item.record_index, item.issue_type)
            if key not in consolidated_map:
                consolidated_map[key] = item

    # 1. Convert malformed records into ERROR anomalies
    malformed_anomalies = [
        _malformed_record_to_anomaly(m) for m in load_result.malformed_records
    ]
    _register_anomalies(malformed_anomalies)

    # 2. Per-record checks: Whitespace/Empty/Missing + Unicode/Burmese diagnostics
    for record in load_result.valid_records:
        ws_findings = check_record_whitespace(record, config=active_config)
        _register_anomalies(ws_findings)

        uni_findings = check_record_unicode(record, config=active_config)
        _register_anomalies(uni_findings)

    # 3. Dataset-wide Duplicate Detection (exact raw match + optional NFC variant check)
    duplicate_groups, dup_anomalies, norm_dup_group_count = detect_duplicates(
        load_result.valid_records, config=active_config
    )
    _register_anomalies(dup_anomalies)

    # 4. Dataset-wide Record-Length Outlier Detection (IQR)
    iqr_bounds, outlier_anomalies = detect_length_outliers(
        load_result.valid_records, config=active_config
    )
    _register_anomalies(outlier_anomalies)

    # 5. Deterministic ordering of all anomalies by (record_index, issue_type.value)
    all_anomalies = sorted(
        consolidated_map.values(),
        key=lambda a: (a.record_index, a.issue_type.value),
    )

    # 6. Compute comprehensive dataset statistics
    stats = compute_dataset_statistics(
        load_result=load_result,
        anomalies=all_anomalies,
        duplicate_groups=duplicate_groups,
        iqr_bounds=iqr_bounds,
        normalized_duplicate_group_count=norm_dup_group_count,
    )

    return AnalysisResult(
        analysis_timestamp_utc=timestamp_utc,
        config=active_config,
        load_result=load_result,
        stats=stats,
        anomalies=all_anomalies,
        duplicate_groups=duplicate_groups,
    )


def inspect_dataset_bytes(
    file_bytes: bytes,
    filename: str,
    config: AnalysisConfig | None = None,
    explicit_format: InputFormat | None = None,
    selected_field: str | None = None,
    csv_delimiter: str | None = None,
    csv_has_header: bool = True,
) -> AnalysisResult:
    """Convenience function to load and analyze a dataset byte stream in one step."""
    active_config = config or AnalysisConfig()
    load_res = load_dataset(
        file_bytes=file_bytes,
        filename=filename,
        config=active_config,
        explicit_format=explicit_format,
        selected_field=selected_field,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
    )
    return analyze_dataset(load_result=load_res, config=active_config)