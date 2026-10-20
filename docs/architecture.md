# KhaingLens v0.1.0 — Architecture & Design Guide

## 1. Architectural Principles
KhaingLens is built around three core architectural tenets:
1. **Decoupled Core Engine (`khainglens/`):** All parsing, quality checks, outlier detection, statistics, and report generation reside in pure Python modules independent of Streamlit. This allows the inspection engine to be tested independently via `pytest` or reused in CLI tools and Jupyter notebooks.
2. **Strict Immutability:** Input records are loaded into frozen Pydantic v2 models (`ParsedRecord`, `MalformedRecord`, `Anomaly`). Diagnostic modules inspect text without ever modifying or deleting the user's data.
3. **Local-First Privacy:** All operations execute locally in machine memory. No dataset contents are transmitted to external AI services or telemetry servers.

## 2. Module Responsibilities
- **`models.py`**: Immutable Pydantic v2 schemas, controlled enums (`InputFormat`, `Severity`, `AnalysisScope`, `AnomalyType`), and safe excerpt builders.
- **`loader.py`**: UTF-8 and UTF-8-BOM stream parsing, CSV header/headerless toggles, delimiter autodetection, malformed row isolation with 1-based line numbers, and streaming record limits (`max_records_limit`).
- **`whitespace_checks.py`**: Distinguishes missing (`None`), empty (`""`), and whitespace-only records; detects leading/trailing whitespace, repeated spaces, and embedded tabs/newlines.
- **`duplicate_detection.py`**: $O(N)$ hash-grouped exact raw-text duplication analysis and optional Unicode NFC canonical normalization variant checks.
- **`unicode_checks.py`**: Inspects control characters, embedded BOMs (`U+FEFF`), zero-width formatting characters (`U+200B`–`U+200D`), unusual spaces, and Burmese-specific orthographic checks (`MYANMAR_ORPHAN_COMBINING`, `MYANMAR_DUPLICATE_DIACRITIC`, `MYANMAR_SUSPECTED_ZAWGYI`) while protecting valid Myanmar Unicode sequences (Kinzi, stacked consonants, medials, logical `U+1031`).
- **`outlier_detection.py`**: Interquartile Range (IQR) record-length outlier detection with small-dataset ($N < 4$) and uniform-length ($\text{IQR} = 0$) safety guards.
- **`statistics.py`**: Aggregates record counts, missing/empty/whitespace-only rates, duplication involvement metrics, Unicode code-point length statistics, pre-aggregated histogram bins, approximate whitespace token counts, and anomaly occurrence sums.
- **`analyzer.py`**: Orchestrates the inspection pipeline, converts malformed records into `MALFORMED_RECORD` errors, enforces the consolidation invariant (at most one anomaly per `(record_index, issue_type)`), and provides $O(1)$ record lookup.
- **`reporting.py`**: RFC 4180-compliant UTF-8 CSV exports (with optional UTF-8 BOM for Microsoft Excel) and structured UTF-8 JSON exports.
- **`app.py`**: Streamlit 6-page responsive user interface adhering to the light-blue WCAG AA design system.