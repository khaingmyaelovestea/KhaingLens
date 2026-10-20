# KhaingLens v0.1.0 — Issue Definitions & Severity Guide

KhaingLens classifies findings into three controlled severity levels:
- **`[!] ERROR`**: Definite structural, parsing, encoding, or missing-field failure.
- **`[▲] WARNING`**: Potentially problematic record requiring human review.
- **`[i] INFO`**: Neutral observation, stylistic pattern, or statistical outlier.

## Complete Glossary of Anomaly Types
1. **`malformed_record` (ERROR):** Row/line failed CSV column parsing, JSONL syntax, or UTF-8 decoding. Isolated separately so valid records are not discarded.
2. **`missing_text_value` (ERROR):** Target text field/column is null or missing.
3. **`empty_string` (WARNING):** Present string of length 0 (`""`).
4. **`whitespace_only` (WARNING):** Non-zero length string consisting exclusively of whitespace characters.
5. **`leading_whitespace` (INFO):** Non-empty text begins with whitespace.
6. **`trailing_whitespace` (INFO):** Non-empty text ends with whitespace.
7. **`repeated_spaces` (INFO):** Contains multiple consecutive ASCII spaces (`U+0020`).
8. **`unusual_tab_or_newline` (WARNING):** Contains embedded tab (`\t`) or newline (`\r`, `\n`) characters.
9. **`exact_duplicate` (INFO):** Raw text is identical to at least one other record.
10. **`normalized_duplicate` (WARNING):** Different raw code-point sequences sharing an identical Unicode NFC canonical form.
11. **`unexpected_control_char` (ERROR):** Contains C0/C1 control characters or Unicode replacement char `U+FFFD`.
12. **`embedded_bom` (WARNING):** Contains Byte Order Mark (`U+FEFF`) mid-record.
13. **`zero_width_character` (WARNING):** Contains invisible formatting marks (`U+200B`, `U+200C`, `U+200D`, `U+2060`, `U+00AD`).
14. **`unusual_unicode_space` (INFO):** Contains non-ASCII space characters (`U+00A0`, `U+3000`).
15. **`unicode_normalization_diff` (INFO):** Raw text differs from its Unicode NFC canonical form.
16. **`myanmar_orphan_combining` (WARNING):** Myanmar combining mark appearing without a preceding base character.
17. **`myanmar_duplicate_diacritic` (WARNING):** Identical Myanmar combining marks typed back-to-back, stacking invisibly on screen.
18. **`myanmar_suspected_zawgyi` (WARNING):** Heuristic match for legacy Zawgyi-One typing patterns (visual-order `U+1031`, trailing `U+1039`, or repurposed ethnic code points).
19. **`length_outlier_short` (INFO):** Record length is strictly below the statistical IQR lower fence.
20. **`length_outlier_long` (INFO):** Record length is strictly above the statistical IQR upper fence.