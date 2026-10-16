"""Interquartile Range (IQR) record-length outlier detection for KhaingLens."""

import statistics

from khainglens.models import (
    AnalysisConfig,
    Anomaly,
    AnomalyType,
    ParsedRecord,
    Severity,
    build_excerpt,
)


def detect_length_outliers(
    records: list[ParsedRecord],
    config: AnalysisConfig | None = None,
) -> tuple[dict[str, float | None], list[Anomaly]]:
    """Identify unusually short or long records using Interquartile Range (IQR).

    Rules:
    - Analyzes non-empty analyzable records (raw_text is not None and raw_text.strip() != "").
    - Requires at least config.min_records_for_iqr (default 4) records.
    - Skips IQR calculation safely if IQR == 0 (all records have identical length).
    - Emits LENGTH_OUTLIER_SHORT or LENGTH_OUTLIER_LONG (Severity.INFO).
    - Returns (bounds_dict, anomalies).
    """
    active_config = config or AnalysisConfig()
    bounds: dict[str, float | None] = {
        "q1": None,
        "q3": None,
        "iqr": None,
        "lower_fence": None,
        "upper_fence": None,
    }
    anomalies: list[Anomaly] = []

    # Filter eligible analyzable non-empty records
    analyzable = [
        r for r in records if r.raw_text is not None and r.raw_text != ""
    ]
    if len(analyzable) < active_config.min_records_for_iqr:
        return bounds, anomalies

    lengths = [len(r.raw_text) for r in analyzable if r.raw_text is not None]
    if not lengths:
        return bounds, anomalies

    try:
        # Compute quartiles using inclusive method
        q_list = statistics.quantiles(lengths, n=4, method="inclusive")
        q1, median, q3 = q_list[0], q_list[1], q_list[2]
    except statistics.StatisticsError:
        return bounds, anomalies

    iqr = q3 - q1
    bounds["q1"] = float(q1)
    bounds["q3"] = float(q3)
    bounds["iqr"] = float(iqr)

    # If IQR is 0 (all records have identical length), statistical fences are undefined/uninformative
    if iqr == 0:
        return bounds, anomalies

    k = active_config.iqr_multiplier
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr

    bounds["lower_fence"] = float(lower_fence)
    bounds["upper_fence"] = float(upper_fence)

    for rec in analyzable:
        assert rec.raw_text is not None
        length = len(rec.raw_text)

        if length < lower_fence:
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anom_{rec.record_index:06d}_{AnomalyType.LENGTH_OUTLIER_SHORT.value}",
                    record_index=rec.record_index,
                    source_line_number=rec.source_line_number,
                    issue_type=AnomalyType.LENGTH_OUTLIER_SHORT,
                    severity=Severity.INFO,
                    title=f"Unusually Short Record Length ({length} code points)",
                    explanation=(
                        f"This record's length ({length} Unicode code points) is strictly below "
                        f"the statistical lower outlier fence ({lower_fence:.1f}, computed as "
                        f"Q1[{q1}] - {k} * IQR[{iqr}]). Statistical outliers are not necessarily errors, "
                        f"but very short records often represent truncated sentences or data artifacts."
                    ),
                    original_excerpt=build_excerpt(rec.raw_text, active_config.excerpt_max_chars),
                    evidence=(
                        f"Record length: {length} cp. Dataset Q1: {q1}, Q3: {q3}, IQR: {iqr}, "
                        f"Lower Fence: {lower_fence:.2f}."
                    ),
                    suggested_action="Review whether this short record is valid or a truncation artifact.",
                    occurrence_count=1,
                    char_position=None,
                    char_positions=[],
                    metadata={
                        "length": length,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_fence": lower_fence,
                        "iqr_multiplier": k,
                    },
                )
            )
        elif length > upper_fence:
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anom_{rec.record_index:06d}_{AnomalyType.LENGTH_OUTLIER_LONG.value}",
                    record_index=rec.record_index,
                    source_line_number=rec.source_line_number,
                    issue_type=AnomalyType.LENGTH_OUTLIER_LONG,
                    severity=Severity.INFO,
                    title=f"Unusually Long Record Length ({length} code points)",
                    explanation=(
                        f"This record's length ({length} Unicode code points) is strictly above "
                        f"the statistical upper outlier fence ({upper_fence:.1f}, computed as "
                        f"Q3[{q3}] + {k} * IQR[{iqr}]). Extremely long records can be multi-document "
                        f"concatenation errors or web scraping boilerplate."
                    ),
                    original_excerpt=build_excerpt(rec.raw_text, active_config.excerpt_max_chars),
                    evidence=(
                        f"Record length: {length} cp. Dataset Q1: {q1}, Q3: {q3}, IQR: {iqr}, "
                        f"Upper Fence: {upper_fence:.2f}."
                    ),
                    suggested_action="Inspect whether this long record represents concatenated text or valid long-form prose.",
                    occurrence_count=1,
                    char_position=None,
                    char_positions=[],
                    metadata={
                        "length": length,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "upper_fence": upper_fence,
                        "iqr_multiplier": k,
                    },
                )
            )

    anomalies.sort(key=lambda a: (a.record_index, a.issue_type.value))
    return bounds, anomalies