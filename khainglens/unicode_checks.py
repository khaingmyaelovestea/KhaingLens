"""Unicode diagnostics and Burmese (Myanmar) script inspection for KhaingLens."""

from collections import Counter
import re
from typing import Any
import unicodedata

from khainglens.models import (
    AnalysisConfig,
    Anomaly,
    AnomalyType,
    ParsedRecord,
    Severity,
    build_excerpt,
)

# Zero-width and invisible formatting characters (excluding U+FEFF which is checked as EMBEDDED_BOM)
ZERO_WIDTH_MAP: dict[int, str] = {
    0x200B: "ZERO WIDTH SPACE (ZWSP)",
    0x200C: "ZERO WIDTH NON-JOINER (ZWNJ)",
    0x200D: "ZERO WIDTH JOINER (ZWJ)",
    0x2060: "WORD JOINER (WJ)",
    0x00AD: "SOFT HYPHEN (SHY)",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x200E: "LEFT-TO-RIGHT MARK (LRM)",
    0x200F: "RIGHT-TO-LEFT MARK (RLM)",
}

# Non-ASCII whitespace characters (excluding standard ASCII space U+0020, \t, \n, \r)
UNUSUAL_SPACE_MAP: dict[int, str] = {
    0x00A0: "NO-BREAK SPACE (NBSP)",
    0x1680: "OGHAM SPACE MARK",
    0x2000: "EN QUAD",
    0x2001: "EM QUAD",
    0x2002: "EN SPACE",
    0x2003: "EM SPACE",
    0x2004: "THREE-PER-EM SPACE",
    0x2005: "FOUR-PER-EM SPACE",
    0x2006: "SIX-PER-EM SPACE",
    0x2007: "FIGURE SPACE",
    0x2008: "PUNCTUATION SPACE",
    0x2009: "THIN SPACE",
    0x200A: "HAIR SPACE",
    0x202F: "NARROW NO-BREAK SPACE",
    0x205F: "MEDIUM MATHEMATICAL SPACE",
    0x3000: "IDEOGRAPHIC SPACE (FULLWIDTH SPACE)",
}

# Myanmar Unicode ranges
# Base consonants & independent vowels that can legitimately host combining marks:
# U+1000..U+102A (Consonants + Independent vowels), U+103F (Great Sa),
# U+1040..U+1049 (Digits - sometimes take case/abbreviation markers like ၂၀၂၆ ခုနှစ် or ၁st-style),
# U+104C..U+104F (Symbols:၌, ၍, ၎, ၏ - note U+104E ၎ regularly takes U+1004 U+103A U+1038 င်း),
# U+1050..U+1055 (Pali/Sanskrit extensions), U+AA60..U+AA76, U+A9E0..U+A9FE.
MYANMAR_COMBINING_MARKS: set[int] = set(range(0x102B, 0x103F)) | set(range(0x1056, 0x105A))

# Identical Myanmar combining marks that should never appear twice in a row
MYANMAR_NON_REPEATABLE_DIACRITICS: set[int] = (
    set(range(0x102B, 0x1039)) | set(range(0x103B, 0x103F)) | {0x103A}
)

# High-confidence Zawgyi-One patterns:
# 1. Visual-order U+1031 (vowel e) at start of string or after whitespace/punctuation,
#    immediately followed by a Myanmar base consonant (U+1000..U+1021).
_ZAWGYI_VISUAL_E_RE = re.compile(r"(?:^|[\s\u104a\u104b(\[\"'“‘])\u1031[\u1000-\u1021]")

# 2. Virama U+1039 followed directly by whitespace or Myanmar punctuation (in Zawgyi, U+1039 is
#    used as visual asat '်' at the end of a syllable, whereas in Unicode U+1039 is an invisible
#    consonant stacker that must be followed by a stacked consonant).
_ZAWGYI_TRAILING_VIRAMA_RE = re.compile(r"\u1039(?:$|[\s\u104a\u104b)\]\"'”’])")

# 3. Code points allocated to Mon (U+1033, U+1034, U+105A) or Shan/Karen extensions (U+1060..U+1097)
#    which Zawgyi-One repurposed for standard Burmese medial ya/ra and stacked consonants.
_ZAWGYI_REPURPOSED_CODEPOINTS: set[int] = {0x1033, 0x1034, 0x105A} | set(range(0x1060, 0x1098))


def _is_valid_myanmar_host(cp: int) -> bool:
    """Return True if code point cp can precede or host a Myanmar combining mark."""
    if 0x1000 <= cp <= 0x103F:
        return True
    if 0x1040 <= cp <= 0x1049:
        # Myanmar digits occasionally precede combining marks in shorthand abbreviations
        return True
    if 0x104C <= cp <= 0x1059:
        return True
    if 0xAA60 <= cp <= 0xAA7F or 0xA9E0 <= cp <= 0xA9FF:
        return True
    return False


def visualize_invisible_characters(text: str) -> str:
    """Render invisible, zero-width, BOM, unusual space, and control characters as visible tokens."""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp == 0xFEFF:
            out.append("[BOM U+FEFF]")
        elif cp in ZERO_WIDTH_MAP:
            short = ZERO_WIDTH_MAP[cp].split("(")[-1].rstrip(")") if "(" in ZERO_WIDTH_MAP[cp] else f"U+{cp:04X}"
            out.append(f"[{short} U+{cp:04X}]")
        elif cp in UNUSUAL_SPACE_MAP:
            out.append(f"[SPACE U+{cp:04X}]")
        elif ch not in ("\t", "\n", "\r") and (
            cp < 0x20 or (0x7F <= cp <= 0x9F) or cp == 0xFFFD
        ):
            out.append(f"[CTRL U+{cp:04X}]")
        else:
            out.append(ch)
    return "".join(out)


def inspect_text_code_points(text: str, max_chars: int = 500) -> list[dict[str, Any]]:
    """Return detailed per-character Unicode metadata for the Unicode Lab inspector."""
    rows: list[dict[str, Any]] = []
    for idx, ch in enumerate(text[:max_chars]):
        cp = ord(ch)
        cat = unicodedata.category(ch)
        name = unicodedata.name(ch, f"UNNAMED OR CONTROL (U+{cp:04X})")
        ccc = unicodedata.combining(ch)
        is_flag = (
            cp == 0xFEFF
            or cp in ZERO_WIDTH_MAP
            or cp in UNUSUAL_SPACE_MAP
            or (ch not in ("\t", "\n", "\r") and (cp < 0x20 or (0x7F <= cp <= 0x9F) or cp == 0xFFFD))
        )
        rows.append(
            {
                "index": idx,
                "character": visualize_invisible_characters(ch),
                "code_point": f"U+{cp:04X}",
                "unicode_name": name,
                "category": cat,
                "combining_class": ccc,
                "is_special_or_invisible": is_flag,
            }
        )
    return rows


def check_record_unicode(
    record: ParsedRecord,
    config: AnalysisConfig | None = None,
) -> list[Anomaly]:
    """Inspect a single ParsedRecord for Unicode and Burmese script anomalies.

    Enforces consolidation: at most one Anomaly per (record_index, AnomalyType).
    Does not mutate record.raw_text.
    """
    if record.is_missing_value or record.raw_text is None or record.raw_text == "":
        return []

    active_config = config or AnalysisConfig()
    raw = record.raw_text
    idx = record.record_index
    line_num = record.source_line_number
    anomalies: list[Anomaly] = []

    # Buckets for single-pass character scan
    control_positions: list[int] = []
    control_counts: Counter[str] = Counter()

    bom_positions: list[int] = []

    zw_positions: list[int] = []
    zw_counts: Counter[str] = Counter()

    unusual_space_positions: list[int] = []
    unusual_space_counts: Counter[str] = Counter()

    orphan_combining_positions: list[int] = []
    orphan_combining_details: list[str] = []

    dup_diacritic_positions: list[int] = []
    dup_diacritic_details: list[str] = []

    zawgyi_repurposed_positions: list[int] = []
    zawgyi_repurposed_counts: Counter[str] = Counter()

    prev_cp: int | None = None

    for pos, ch in enumerate(raw):
        cp = ord(ch)

        # 1. Embedded BOM (U+FEFF)
        if cp == 0xFEFF:
            bom_positions.append(pos)

        # 2. Zero-width characters
        elif cp in ZERO_WIDTH_MAP:
            zw_positions.append(pos)
            zw_counts[f"U+{cp:04X} {ZERO_WIDTH_MAP[cp]}"] += 1

        # 3. Unusual Unicode spaces
        elif cp in UNUSUAL_SPACE_MAP:
            unusual_space_positions.append(pos)
            unusual_space_counts[f"U+{cp:04X} {UNUSUAL_SPACE_MAP[cp]}"] += 1

        # 4. Unexpected C0/C1 control characters or U+FFFD replacement character
        elif ch not in ("\t", "\n", "\r") and (
            cp < 0x20 or (0x7F <= cp <= 0x9F) or cp == 0xFFFD or unicodedata.category(ch) == "Cc"
        ):
            control_positions.append(pos)
            cname = unicodedata.name(ch, "CONTROL CHARACTER")
            control_counts[f"U+{cp:04X} ({cname})"] += 1

        # 5. Myanmar Orphan Combining Mark check
        if cp in MYANMAR_COMBINING_MARKS:
            if prev_cp is None or not _is_valid_myanmar_host(prev_cp):
                orphan_combining_positions.append(pos)
                mname = unicodedata.name(ch, "MYANMAR MARK")
                prev_desc = "start of string" if prev_cp is None else f"U+{prev_cp:04X} ('{chr(prev_cp)}')"
                orphan_combining_details.append(
                    f"U+{cp:04X} ({mname}) at index {pos} after {prev_desc}"
                )

        # 6. Myanmar Duplicate Adjacent Diacritic check
        if cp in MYANMAR_NON_REPEATABLE_DIACRITICS and prev_cp == cp:
            dup_diacritic_positions.append(pos)
            mname = unicodedata.name(ch, "MYANMAR MARK")
            dup_diacritic_details.append(
                f"Repeated U+{cp:04X} ({mname}) at indices {pos - 1}..{pos}"
            )

        # 7. Zawgyi-repurposed code points (U+1033, U+1034, U+105A, U+1060..U+1097)
        if cp in _ZAWGYI_REPURPOSED_CODEPOINTS:
            zawgyi_repurposed_positions.append(pos)
            mname = unicodedata.name(ch, "MYANMAR EXTENSION")
            zawgyi_repurposed_counts[f"U+{cp:04X} ({mname})"] += 1

        prev_cp = cp

    # Emit consolidated Anomaly 1: UNEXPECTED_CONTROL_CHAR
    if control_positions:
        first_pos = control_positions[0]
        breakdown = ", ".join(f"{c}x {k}" for k, c in control_counts.items())
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.UNEXPECTED_CONTROL_CHAR.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.UNEXPECTED_CONTROL_CHAR,
                severity=Severity.ERROR,
                title=f"Unexpected Control or Replacement Character ({len(control_positions)} occurrence(s))",
                explanation=(
                    "The record contains non-printable C0/C1 control characters or the Unicode "
                    "replacement character (U+FFFD), which typically indicates binary corruption, "
                    "prior lossy decoding, or terminal escape sequences."
                ),
                original_excerpt=build_excerpt(
                    visualize_invisible_characters(raw),
                    active_config.excerpt_max_chars,
                    focus_pos=first_pos,
                ),
                evidence=(
                    f"{len(control_positions)} control/replacement character(s) at positions "
                    f"{control_positions[:15]}: {breakdown}"
                ),
                suggested_action=(
                    "Remove non-printable control characters and verify upstream character encoding."
                ),
                occurrence_count=len(control_positions),
                char_position=first_pos,
                char_positions=control_positions,
                metadata={
                    "counts_by_code_point": dict(control_counts),
                    "positions": control_positions,
                },
            )
        )

    # Emit consolidated Anomaly 2: EMBEDDED_BOM
    if bom_positions:
        first_pos = bom_positions[0]
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.EMBEDDED_BOM.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.EMBEDDED_BOM,
                severity=Severity.WARNING,
                title=f"Embedded Byte-Order Mark U+FEFF ({len(bom_positions)} occurrence(s))",
                explanation=(
                    "A Unicode Byte-Order Mark (U+FEFF / ZERO WIDTH NO-BREAK SPACE) appears inside "
                    "the record text. This commonly happens when multiple UTF-8-BOM files are "
                    "concatenated without stripping headers, corrupting token boundaries."
                ),
                original_excerpt=build_excerpt(
                    visualize_invisible_characters(raw),
                    active_config.excerpt_max_chars,
                    focus_pos=first_pos,
                ),
                evidence=f"U+FEFF detected at character index(es) {bom_positions[:15]}.",
                suggested_action="Strip embedded U+FEFF characters during corpus cleaning.",
                occurrence_count=len(bom_positions),
                char_position=first_pos,
                char_positions=bom_positions,
                metadata={"code_point": "U+FEFF", "positions": bom_positions},
            )
        )

    # Emit consolidated Anomaly 3: ZERO_WIDTH_CHARACTER
    if zw_positions:
        first_pos = zw_positions[0]
        breakdown = ", ".join(f"{c}x {k}" for k, c in zw_counts.items())
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.ZERO_WIDTH_CHARACTER.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.ZERO_WIDTH_CHARACTER,
                severity=Severity.WARNING,
                title=f"Invisible Zero-Width Characters ({len(zw_positions)} occurrence(s))",
                explanation=(
                    "The record contains invisible zero-width characters (such as U+200B ZERO WIDTH SPACE "
                    "or U+200C ZWNJ). Note for Burmese NLP: U+200B is sometimes intentionally inserted "
                    "as a syllable/word boundary marker, but in web-scraped datasets it is frequently an "
                    "unintended copy-paste artifact that fragments subword tokenization."
                ),
                original_excerpt=build_excerpt(
                    visualize_invisible_characters(raw),
                    active_config.excerpt_max_chars,
                    focus_pos=first_pos,
                ),
                evidence=(
                    f"{len(zw_positions)} zero-width character(s) at positions "
                    f"{zw_positions[:15]}: {breakdown}"
                ),
                suggested_action=(
                    "Inspect in Unicode Lab to confirm whether zero-width characters are intentional "
                    "segmentation markers or scraping artifacts."
                ),
                occurrence_count=len(zw_positions),
                char_position=first_pos,
                char_positions=zw_positions,
                metadata={
                    "counts_by_code_point": dict(zw_counts),
                    "positions": zw_positions,
                    "visualized_excerpt": build_excerpt(
                        visualize_invisible_characters(raw), 200, focus_pos=first_pos
                    ),
                },
            )
        )

    # Emit consolidated Anomaly 4: UNUSUAL_UNICODE_SPACE
    if unusual_space_positions:
        first_pos = unusual_space_positions[0]
        breakdown = ", ".join(f"{c}x {k}" for k, c in unusual_space_counts.items())
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.UNUSUAL_UNICODE_SPACE.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.UNUSUAL_UNICODE_SPACE,
                severity=Severity.INFO,
                title=f"Non-Standard Unicode Whitespace ({len(unusual_space_positions)} occurrence(s))",
                explanation=(
                    "The record contains non-ASCII whitespace characters such as U+00A0 (NO-BREAK SPACE) "
                    "or U+3000 (IDEOGRAPHIC SPACE). Visually identical to standard spaces (U+0020), "
                    "they can prevent exact phrase matching if not normalized."
                ),
                original_excerpt=build_excerpt(
                    visualize_invisible_characters(raw),
                    active_config.excerpt_max_chars,
                    focus_pos=first_pos,
                ),
                evidence=(
                    f"{len(unusual_space_positions)} unusual space(s) at positions "
                    f"{unusual_space_positions[:15]}: {breakdown}"
                ),
                suggested_action=(
                    "Review whether non-breaking or fullwidth spaces should be converted to standard ASCII space (U+0020)."
                ),
                occurrence_count=len(unusual_space_positions),
                char_position=first_pos,
                char_positions=unusual_space_positions,
                metadata={
                    "counts_by_code_point": dict(unusual_space_counts),
                    "positions": unusual_space_positions,
                },
            )
        )

    # Emit consolidated Anomaly 5: UNICODE_NORMALIZATION_DIFF (NFC check)
    nfc_text = unicodedata.normalize("NFC", raw)
    if nfc_text != raw:
        # Find the first index where raw and nfc_text diverge
        diff_pos = 0
        min_len = min(len(raw), len(nfc_text))
        while diff_pos < min_len and raw[diff_pos] == nfc_text[diff_pos]:
            diff_pos += 1

        raw_span = raw[max(0, diff_pos - 2) : min(len(raw), diff_pos + 6)]
        nfc_span = nfc_text[max(0, diff_pos - 2) : min(len(nfc_text), diff_pos + 6)]
        raw_cps = " ".join(f"U+{ord(c):04X}" for c in raw_span)
        nfc_cps = " ".join(f"U+{ord(c):04X}" for c in nfc_span)

        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.UNICODE_NORMALIZATION_DIFF.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.UNICODE_NORMALIZATION_DIFF,
                severity=Severity.INFO,
                title="Non-NFC Unicode Sequence Detected",
                explanation=(
                    "This record's raw code-point sequence differs from its Unicode Normalization "
                    "Form C (NFC) representation. In Burmese, this frequently occurs when combining "
                    "marks with different Canonical Combining Classes (e.g., U+1037 dot below, CCC=7, "
                    "and U+103A asat, CCC=9) are stored in non-canonical order. Note: Standard Unicode "
                    "NFC only reorders marks with distinct non-zero CCC values."
                ),
                original_excerpt=build_excerpt(raw, active_config.excerpt_max_chars, focus_pos=diff_pos),
                evidence=(
                    f"First divergence near index {diff_pos}: Raw [{raw_cps}] vs NFC [{nfc_cps}]."
                ),
                suggested_action=(
                    "Compare raw vs. NFC code points in Unicode Lab if consistent canonical ordering is required."
                ),
                occurrence_count=1,
                char_position=diff_pos,
                char_positions=[diff_pos],
                metadata={
                    "first_diff_position": diff_pos,
                    "raw_length": len(raw),
                    "nfc_length": len(nfc_text),
                    "raw_span_code_points": raw_cps,
                    "nfc_span_code_points": nfc_cps,
                },
            )
        )

    # Emit consolidated Anomaly 6: MYANMAR_ORPHAN_COMBINING
    if orphan_combining_positions:
        first_pos = orphan_combining_positions[0]
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.MYANMAR_ORPHAN_COMBINING.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.MYANMAR_ORPHAN_COMBINING,
                severity=Severity.WARNING,
                title=f"Orphaned Myanmar Combining Mark ({len(orphan_combining_positions)} occurrence(s))",
                explanation=(
                    "A Myanmar dependent vowel, medial, virama, or tone mark (U+102B..U+103E) appears "
                    "at the start of the text or immediately following whitespace, punctuation, or non-Myanmar "
                    "characters without a valid Myanmar base consonant. This occurs from broken string slicing "
                    "or visual-order Zawgyi typing (e.g., typing U+1031 before a consonant)."
                ),
                original_excerpt=build_excerpt(raw, active_config.excerpt_max_chars, focus_pos=first_pos),
                evidence="; ".join(orphan_combining_details[:8]),
                suggested_action=(
                    "Inspect the syllable structure around the flagged position for broken slicing or Zawgyi input."
                ),
                occurrence_count=len(orphan_combining_positions),
                char_position=first_pos,
                char_positions=orphan_combining_positions,
                metadata={
                    "positions": orphan_combining_positions,
                    "details": orphan_combining_details[:20],
                },
            )
        )

    # Emit consolidated Anomaly 7: MYANMAR_DUPLICATE_DIACRITIC
    if dup_diacritic_positions:
        first_pos = dup_diacritic_positions[0]
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.MYANMAR_DUPLICATE_DIACRITIC.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.MYANMAR_DUPLICATE_DIACRITIC,
                severity=Severity.WARNING,
                title=f"Duplicate Adjacent Myanmar Diacritic ({len(dup_diacritic_positions)} occurrence(s))",
                explanation=(
                    "Two identical Myanmar combining marks (such as double vowel signs, medials, asat U+103A, "
                    "or dot-below U+1037) appear immediately adjacent to each other. Font rendering engines "
                    "stack duplicate marks in the exact same screen position, making this typo invisible to "
                    "human readers while corrupting character-level and subword NLP models."
                ),
                original_excerpt=build_excerpt(raw, active_config.excerpt_max_chars, focus_pos=first_pos),
                evidence="; ".join(dup_diacritic_details[:8]),
                suggested_action="Inspect and deduplicate the repeated Myanmar combining mark.",
                occurrence_count=len(dup_diacritic_positions),
                char_position=first_pos,
                char_positions=dup_diacritic_positions,
                metadata={
                    "positions": dup_diacritic_positions,
                    "details": dup_diacritic_details[:20],
                },
            )
        )

    # Emit consolidated Anomaly 8: MYANMAR_SUSPECTED_ZAWGYI
    zawgyi_positions_set: set[int] = set(zawgyi_repurposed_positions)
    zawgyi_reasons: list[str] = []

    for m in _ZAWGYI_VISUAL_E_RE.finditer(raw):
        # Position of U+1031 inside the match
        e_offset = m.group(0).find("\u1031")
        e_pos = m.start() + (e_offset if e_offset >= 0 else 0)
        zawgyi_positions_set.add(e_pos)
        zawgyi_reasons.append(
            f"Visual-order vowel U+1031 preceding base consonant at index {e_pos}"
        )

    for m in _ZAWGYI_TRAILING_VIRAMA_RE.finditer(raw):
        v_pos = m.start()
        zawgyi_positions_set.add(v_pos)
        zawgyi_reasons.append(
            f"Virama U+1039 used at syllable end before whitespace/punctuation at index {v_pos}"
        )

    if zawgyi_repurposed_counts:
        repurposed_str = ", ".join(f"{c}x {k}" for k, c in zawgyi_repurposed_counts.items())
        zawgyi_reasons.append(
            f"Code points commonly repurposed in Zawgyi-One (or used in Mon/Shan/Karen ethnic scripts): {repurposed_str}"
        )

    if zawgyi_positions_set:
        sorted_z_positions = sorted(zawgyi_positions_set)
        first_pos = sorted_z_positions[0]
        anomalies.append(
            Anomaly(
                anomaly_id=f"anom_{idx:06d}_{AnomalyType.MYANMAR_SUSPECTED_ZAWGYI.value}",
                record_index=idx,
                source_line_number=line_num,
                issue_type=AnomalyType.MYANMAR_SUSPECTED_ZAWGYI,
                severity=Severity.WARNING,
                title=f"Suspected Zawgyi-One or Ethnic Script Sequence ({len(sorted_z_positions)} indicator(s))",
                explanation=(
                    "Heuristic check detected code-point patterns typical of legacy Zawgyi-One font encoding "
                    "(such as U+1031 preceding a consonant in visual order, trailing U+1039 virama, or "
                    "code points U+1033/U+1034/U+105A/U+1060..U+1097). Important note: U+1033, U+1034, and "
                    "U+105A are valid in Mon language Unicode texts, and U+1060..U+1097 are valid in Karen/Shan "
                    "Unicode texts. Verify whether this dataset is standard Burmese (containing Zawgyi leakage) "
                    "or a multilingual ethnic-script corpus."
                ),
                original_excerpt=build_excerpt(raw, active_config.excerpt_max_chars, focus_pos=first_pos),
                evidence="; ".join(zawgyi_reasons[:8]),
                suggested_action=(
                    "If this is a Burmese dataset, convert Zawgyi-One records to Unicode 5.1+ using a verified "
                    "transducer (e.g., Rabbit or ICU my-Qaag-t-my) after human review."
                ),
                occurrence_count=len(sorted_z_positions),
                char_position=first_pos,
                char_positions=sorted_z_positions,
                metadata={
                    "positions": sorted_z_positions,
                    "heuristic_indicators": zawgyi_reasons[:15],
                },
            )
        )

    return anomalies