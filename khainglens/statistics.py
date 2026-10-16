"""Comprehensive dataset statistics and metric aggregation for KhaingLens."""

from collections import Counter, defaultdict
import statistics

from khainglens.models import (
    Anomaly,
    AnomalyType,
    DatasetStats,
    DuplicateGroup,
    HistogramBin,
    LoadResult,
)


def compute_dataset_statistics(
    load_result: LoadResult,
    anomalies: list[Anomaly],
    duplicate_groups: list[DuplicateGroup],
    iqr_bounds: dict[str, float | None],
    normalized_duplicate_group_count: int | None = None,
) -> DatasetStats:
    """Aggregate dataset statistics, anomaly metrics, rates, and length histograms."""
    valid_recs = load_result.valid_records
    malformed_recs = load_result.malformed_records

    total_raw = load_result.analyzed_raw_records
    n_parsed = len(valid_recs)
    n_malformed = len(malformed_recs)

    # Missing vs Empty vs Whitespace-only counts
    n_missing = sum(1 for r in valid_recs if r.is_missing_value or r.raw_text is None)
    n_empty = sum(1 for r in valid_recs if not r.is_missing_value and r.raw_text == "")
    n_ws_only = sum(
        1
        for r in valid_recs
        if not r.is_missing_value
        and r.raw_text is not None
        and r.raw_text != ""
        and r.raw_text.strip() == ""
    )
    total_blank = n_missing + n_empty + n_ws_only
    n_analyzable = n_parsed - total_blank

    # Duplication metrics over non-empty analyzable records
    analyzable_texts: list[str] = [
        r.raw_text
        for r in valid_recs
        if not r.is_missing_value and r.raw_text is not None and r.raw_text.strip() != ""
    ]
    unique_texts_count = len(set(analyzable_texts))
    dup_group_count = len(duplicate_groups)
    total_recs_in_dups = sum(g.occurrence_count for g in duplicate_groups)
    surplus_dups = sum(g.surplus_count for g in duplicate_groups)

    # Count distinct normalized duplicate groups
    if normalized_duplicate_group_count is not None:
        norm_dup_groups = normalized_duplicate_group_count
    else:
        nfc_group_ids = {
            a.metadata.get("nfc_group_id")
            for a in anomalies
            if a.issue_type == AnomalyType.NORMALIZED_DUPLICATE
            and "nfc_group_id" in a.metadata
        }
        norm_dup_groups = len(nfc_group_ids)

    # Length statistics over analyzable records (Unicode code points)
    lengths = [len(t) for t in analyzable_texts]
    min_len: int | None = None
    max_len: int | None = None
    mean_len: float | None = None
    median_len: float | None = None
    q1_len = iqr_bounds.get("q1")
    q3_len = iqr_bounds.get("q3")
    iqr_len = iqr_bounds.get("iqr")
    lower_fence = iqr_bounds.get("lower_fence")
    upper_fence = iqr_bounds.get("upper_fence")
    histogram: list[HistogramBin] = []

    if lengths:
        min_len = min(lengths)
        max_len = max(lengths)
        mean_len = float(statistics.mean(lengths))
        median_len = float(statistics.median(lengths))

        # Build pre-aggregated histogram bins (at most 25 bins)
        if min_len == max_len:
            histogram = [
                HistogramBin(
                    bin_start=float(min_len),
                    bin_end=float(min_len + 1),
                    count=len(lengths),
                )
            ]
        else:
            num_bins = min(25, max(5, len(lengths) // 10))
            bin_width = (max_len - min_len) / num_bins
            bins_data = [0] * num_bins
            for length_val in lengths:
                b_idx = int((length_val - min_len) / bin_width)
                if b_idx >= num_bins:
                    b_idx = num_bins - 1
                bins_data[b_idx] += 1

            for i in range(num_bins):
                start = min_len + i * bin_width
                end = min_len + (i + 1) * bin_width
                histogram.append(
                    HistogramBin(
                        bin_start=round(start, 2),
                        bin_end=round(end, 2),
                        count=bins_data[i],
                    )
                )

    # Approximate whitespace token counts (explicitly labeled for Burmese transparency)
    token_counts = [len(t.split()) for t in analyzable_texts]
    mean_tokens = float(statistics.mean(token_counts)) if token_counts else None
    max_tokens = max(token_counts) if token_counts else None

    # Anomaly & affected-record metrics
    total_findings = len(anomalies)
    total_occurrence_sum = sum(a.occurrence_count for a in anomalies)
    records_with_anomaly = len({a.record_index for a in anomalies})

    counts_by_type: Counter[str] = Counter()
    occurrence_sums_by_type: Counter[str] = Counter()
    affected_recs_by_type: dict[str, set[int]] = defaultdict(set)

    counts_by_severity: Counter[str] = Counter()
    affected_recs_by_severity: dict[str, set[int]] = defaultdict(set)

    for a in anomalies:
        t_key = a.issue_type.value
        s_key = a.severity.value

        counts_by_type[t_key] += 1
        occurrence_sums_by_type[t_key] += a.occurrence_count
        affected_recs_by_type[t_key].add(a.record_index)

        counts_by_severity[s_key] += 1
        affected_recs_by_severity[s_key].add(a.record_index)

    affected_by_type_final = {k: len(v) for k, v in affected_recs_by_type.items()}
    affected_by_sev_final = {k: len(v) for k, v in affected_recs_by_severity.items()}

    # Rates (0.0 to 1.0, None when denominator is 0)
    malformed_rate = (n_malformed / total_raw) if total_raw > 0 else None
    empty_missing_rate = (total_blank / n_parsed) if n_parsed > 0 else None
    dup_involvement_rate = (total_recs_in_dups / n_analyzable) if n_analyzable > 0 else None
    surplus_dup_rate = (surplus_dups / n_analyzable) if n_analyzable > 0 else None
    affected_record_rate = (records_with_anomaly / total_raw) if total_raw > 0 else None

    return DatasetStats(
        analysis_scope=load_result.analysis_scope,
        truncated_at_limit=load_result.truncated_at_limit,
        max_records_limit_applied=load_result.max_records_limit_applied,
        analyzed_raw_records=total_raw,
        successfully_parsed_records=n_parsed,
        malformed_record_count=n_malformed,
        missing_value_record_count=n_missing,
        empty_string_record_count=n_empty,
        whitespace_only_record_count=n_ws_only,
        total_empty_or_missing_records=total_blank,
        non_empty_analyzable_records=n_analyzable,
        unique_non_empty_texts=unique_texts_count,
        duplicate_group_count=dup_group_count,
        total_records_in_duplicate_groups=total_recs_in_dups,
        surplus_duplicate_occurrences=surplus_dups,
        normalized_duplicate_group_count=norm_dup_groups,
        length_unit="unicode_code_points",
        min_length=min_len,
        max_length=max_len,
        mean_length=mean_len,
        median_length=median_len,
        q1_length=q1_len,
        q3_length=q3_len,
        iqr_length=iqr_len,
        lower_outlier_fence=lower_fence,
        upper_outlier_fence=upper_fence,
        length_histogram=histogram,
        mean_approx_whitespace_tokens=mean_tokens,
        max_approx_whitespace_tokens=max_tokens,
        total_anomaly_findings=total_findings,
        total_in_record_anomaly_occurrences=total_occurrence_sum,
        records_with_any_anomaly=records_with_anomaly,
        anomaly_counts_by_type=dict(counts_by_type),
        occurrence_sums_by_type=dict(occurrence_sums_by_type),
        affected_records_by_type=affected_by_type_final,
        anomaly_counts_by_severity=dict(counts_by_severity),
        affected_records_by_severity=affected_by_sev_final,
        malformed_record_rate=malformed_rate,
        empty_or_missing_rate=empty_missing_rate,
        duplicate_involvement_rate=dup_involvement_rate,
        surplus_duplicate_rate=surplus_dup_rate,
        affected_record_rate=affected_record_rate,
    )