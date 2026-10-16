"""Exact and canonical-normalized duplicate detection for KhaingLens."""

from collections import defaultdict
import unicodedata

from khainglens.models import (
    AnalysisConfig,
    Anomaly,
    AnomalyType,
    DuplicateGroup,
    ParsedRecord,
    Severity,
    build_excerpt,
)


def _format_code_points(text: str, max_chars: int = 25) -> str:
    """Format the first max_chars code points of a string for diagnostic comparison."""
    points = [f"U+{ord(ch):04X}" for ch in text[:max_chars]]
    suffix = " ..." if len(text) > max_chars else ""
    return " ".join(points) + suffix


def detect_duplicates(
    records: list[ParsedRecord],
    config: AnalysisConfig | None = None,
) -> tuple[list[DuplicateGroup], list[Anomaly], int]:
    """Detect exact raw-text duplicates and optional NFC canonical duplicates.

    Rules:
    - Excludes missing (None), empty (""), and whitespace-only strings.
    - Exact duplicates compare original unmutated raw_text (s1 == s2).
    - Emits one consolidated Anomaly per (record_index, AnomalyType) with occurrence_count=1
      so dataset occurrence sums equal the exact number of duplicate records.
    - Returns (duplicate_groups, anomalies, normalized_duplicate_group_count).
    """
    active_config = config or AnalysisConfig()

    # 1. Group eligible non-empty records by exact raw_text in O(N)
    raw_buckets: dict[str, list[ParsedRecord]] = defaultdict(list)
    for rec in records:
        if rec.is_missing_value or rec.raw_text is None:
            continue
        if rec.raw_text.strip() == "":
            continue
        raw_buckets[rec.raw_text].append(rec)

    duplicate_groups: list[DuplicateGroup] = []
    anomalies: list[Anomaly] = []
    group_counter = 0

    for raw_text, group_records in raw_buckets.items():
        if len(group_records) < 2:
            continue

        group_counter += 1
        indices = [r.record_index for r in group_records]
        occ_count = len(indices)
        surplus = occ_count - 1
        excerpt = build_excerpt(raw_text, active_config.excerpt_max_chars)

        dup_group = DuplicateGroup(
            group_id=group_counter,
            comparison_basis="raw_exact_match",
            representative_excerpt=excerpt,
            record_indices=indices,
            occurrence_count=occ_count,
            surplus_count=surplus,
        )
        duplicate_groups.append(dup_group)

        for pos_in_group, rec in enumerate(group_records):
            other_indices = [i for i in indices if i != rec.record_index]
            role_label = (
                "first occurrence"
                if pos_in_group == 0
                else f"duplicate copy #{pos_in_group} of {surplus}"
            )
            anomalies.append(
                Anomaly(
                    anomaly_id=f"anom_{rec.record_index:06d}_{AnomalyType.EXACT_DUPLICATE.value}",
                    record_index=rec.record_index,
                    source_line_number=rec.source_line_number,
                    issue_type=AnomalyType.EXACT_DUPLICATE,
                    severity=Severity.INFO,
                    title=f"Exact Duplicate Record (Group #{group_counter}, {occ_count} total)",
                    explanation=(
                        f"This record's raw text is character-for-character identical to {surplus} "
                        f"other record(s) in the analyzed dataset ({role_label}). Exact duplicates "
                        "are not automatically errors (e.g., natural frequency in sentiment corpora), "
                        "but unexpected repetition can cause train/test leakage or model memorization."
                    ),
                    original_excerpt=excerpt,
                    evidence=(
                        f"Exact raw text match across {occ_count} records at indices "
                        f"{indices[:15]}{'...' if len(indices) > 15 else ''}."
                    ),
                    suggested_action=(
                        "Verify whether repeated examples are intentional or should be deduplicated before splitting."
                    ),
                    occurrence_count=1,
                    char_position=None,
                    char_positions=[],
                    metadata={
                        "group_id": group_counter,
                        "comparison_basis": "raw_exact_match",
                        "group_occurrence_count": occ_count,
                        "group_surplus_count": surplus,
                        "is_first_occurrence": pos_in_group == 0,
                        "matching_record_indices": indices,
                        "other_record_indices": other_indices[:50],
                    },
                )
            )

    # 2. Check for NFC Normalization Duplicates (different raw strings that normalize to identical NFC)
    normalized_group_count = 0
    if active_config.detect_normalized_duplicates:
        nfc_to_raw_variants: dict[str, list[str]] = defaultdict(list)
        for distinct_raw in raw_buckets.keys():
            nfc_form = unicodedata.normalize("NFC", distinct_raw)
            nfc_to_raw_variants[nfc_form].append(distinct_raw)

        for nfc_form, raw_variants in nfc_to_raw_variants.items():
            if len(raw_variants) < 2:
                continue

            normalized_group_count += 1
            all_affected_records: list[ParsedRecord] = []
            for variant in raw_variants:
                all_affected_records.extend(raw_buckets[variant])
            all_affected_records.sort(key=lambda r: r.record_index)
            affected_indices = [r.record_index for r in all_affected_records]

            variant_summaries = [
                f"Variant {i + 1} ({len(raw_buckets[v])} rec(s)): [{_format_code_points(v, 12)}]"
                for i, v in enumerate(raw_variants[:4])
            ]
            variants_evidence = "; ".join(variant_summaries)

            for rec in all_affected_records:
                assert rec.raw_text is not None
                anomalies.append(
                    Anomaly(
                        anomaly_id=f"anom_{rec.record_index:06d}_{AnomalyType.NORMALIZED_DUPLICATE.value}",
                        record_index=rec.record_index,
                        source_line_number=rec.source_line_number,
                        issue_type=AnomalyType.NORMALIZED_DUPLICATE,
                        severity=Severity.WARNING,
                        title=(
                            f"Unicode Normalization Duplicate ({len(raw_variants)} distinct raw variants)"
                        ),
                        explanation=(
                            "This record shares an identical Unicode NFC canonical form with other record(s) "
                            "that have different raw code-point sequences. In Burmese and accented scripts, "
                            "this happens when combining marks or precomposed characters are stored in "
                            "different orders—rendering identically on screen while failing exact string deduplication."
                        ),
                        original_excerpt=build_excerpt(
                            rec.raw_text, active_config.excerpt_max_chars
                        ),
                        evidence=(
                            f"Matches under Unicode NFC across {len(affected_indices)} records "
                            f"(indices {affected_indices[:10]}), spanning {len(raw_variants)} distinct "
                            f"raw code-point sequences: {variants_evidence}"
                        ),
                        suggested_action=(
                            "Inspect the code-point sequences in Unicode Lab to decide if NFC normalization is appropriate."
                        ),
                        occurrence_count=1,
                        char_position=None,
                        char_positions=[],
                        metadata={
                            "nfc_group_id": normalized_group_count,
                            "distinct_raw_variant_count": len(raw_variants),
                            "affected_record_indices": affected_indices,
                            "record_raw_code_points": _format_code_points(rec.raw_text, 30),
                            "nfc_code_points": _format_code_points(nfc_form, 30),
                        },
                    )
                )

    anomalies.sort(key=lambda a: (a.record_index, a.issue_type.value))
    return duplicate_groups, anomalies, normalized_group_count