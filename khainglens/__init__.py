"""KhaingLens: 1-Line Python Library & Toolkit for English and Burmese Dataset Quality Inspection."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

# Define package metadata BEFORE importing submodules to prevent circular imports
__version__ = "0.1.0"
__app_name__ = "KhaingLens"

from khainglens.analyzer import analyze_dataset, inspect_dataset_bytes
from khainglens.loader import load_dataset
from khainglens.models import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
    Anomaly,
    AnomalyType,
    DatasetStats,
    DuplicateGroup,
    InputFormat,
    LoadResult,
    ParsedRecord,
    Severity,
)
from khainglens.reporting import generate_csv_report, generate_json_report
from khainglens.unicode_checks import (
    inspect_text_code_points,
    visualize_invisible_characters,
)


def inspect(
    source: str | Path | list[str] | pd.DataFrame | pd.Series,
    field: str | None = None,
    *,
    csv_has_header: bool = True,
    csv_delimiter: str | None = None,
    config: AnalysisConfig | None = None,
    print_summary: bool = True,
) -> AnalysisResult:
    """Inspect a dataset file (.csv, .jsonl, .txt), Pandas DataFrame/Series, or list of strings in 1 line.

    Examples:
        >>> import khainglens as kl
        >>> res = kl.inspect("sample_data/sample_burmese.csv")
        >>> res = kl.inspect(["မင်္ဂလာပါ", "မင်္ဂလာပါ", "မြန်\\u200bမာစာ"])
        >>> res = kl.inspect(df, field="text")
    """
    cfg = config or AnalysisConfig()

    if isinstance(source, (str, Path)):
        path = Path(source)
        file_bytes = path.read_bytes()
        result = inspect_dataset_bytes(
            file_bytes=file_bytes,
            filename=path.name,
            config=cfg,
            selected_field=field,
            csv_delimiter=csv_delimiter,
            csv_has_header=csv_has_header,
        )
    elif isinstance(source, pd.DataFrame):
        target_col = field or (str(source.columns[0]) if len(source.columns) > 0 else "text")
        if target_col not in source.columns:
            raise ValueError(
                f"Column '{target_col}' not found in DataFrame columns: {list(source.columns)}"
            )
        records = [
            ParsedRecord(
                record_index=idx,
                source_line_number=idx + 1,
                raw_text=None if pd.isna(val) else str(val),
            )
            for idx, val in enumerate(source[target_col].tolist()[: cfg.max_records_limit])
        ]
        truncated = len(source) > cfg.max_records_limit
        load_res = LoadResult(
            filename="DataFrame",
            file_format=InputFormat.CSV,
            encoding_used="utf-8",
            selected_field=target_col,
            analyzed_raw_records=len(records),
            truncated_at_limit=truncated,
            max_records_limit_applied=cfg.max_records_limit,
            analysis_scope=(
                AnalysisScope.TRUNCATED_SUBSET if truncated else AnalysisScope.FULL_DATASET
            ),
            valid_records=records,
        )
        result = analyze_dataset(load_res, config=cfg)
    elif isinstance(source, (list, pd.Series)):
        items = source.tolist() if isinstance(source, pd.Series) else list(source)
        records = [
            ParsedRecord(
                record_index=idx,
                source_line_number=idx + 1,
                raw_text=(
                    None
                    if val is None or (isinstance(val, float) and pd.isna(val))
                    else str(val)
                ),
            )
            for idx, val in enumerate(items[: cfg.max_records_limit])
        ]
        truncated = len(items) > cfg.max_records_limit
        load_res = LoadResult(
            filename="in_memory_sequence",
            file_format=InputFormat.TXT,
            encoding_used="utf-8",
            selected_field="text",
            analyzed_raw_records=len(records),
            truncated_at_limit=truncated,
            max_records_limit_applied=cfg.max_records_limit,
            analysis_scope=(
                AnalysisScope.TRUNCATED_SUBSET if truncated else AnalysisScope.FULL_DATASET
            ),
            valid_records=records,
        )
        result = analyze_dataset(load_res, config=cfg)
    else:
        raise TypeError(
            "source must be a file path (str/Path), pandas DataFrame/Series, or list of strings."
        )

    if print_summary:
        s = result.stats
        print(
            f"[KhaingLens] Inspected {s.analyzed_raw_records:,} record(s) from '{result.load_result.filename}' | "
            f"Affected Records: {s.records_with_any_anomaly:,} ({(s.affected_record_rate or 0)*100:.1f}%) | "
            f"Findings: {s.total_anomaly_findings:,} (Errors: {s.anomaly_counts_by_severity.get('error', 0)}, "
            f"Warnings: {s.anomaly_counts_by_severity.get('warning', 0)}, "
            f"Info: {s.anomaly_counts_by_severity.get('info', 0)}) | "
            f"Exact Dup Groups: {s.duplicate_group_count:,}"
        )

    return result


def to_dataframe(result: AnalysisResult) -> pd.DataFrame:
    """Convert all detected anomalies in an AnalysisResult into a Pandas DataFrame in 1 line."""
    if not result.anomalies:
        return pd.DataFrame(
            columns=[
                "record_index",
                "source_line_number",
                "severity",
                "issue_type",
                "occurrence_count",
                "char_position",
                "title",
                "evidence",
                "original_excerpt",
            ]
        )
    return pd.DataFrame(
        [
            {
                "record_index": a.record_index,
                "source_line_number": a.source_line_number,
                "severity": a.severity.value,
                "issue_type": a.issue_type.value,
                "occurrence_count": a.occurrence_count,
                "char_position": a.char_position,
                "title": a.title,
                "evidence": a.evidence,
                "original_excerpt": a.original_excerpt,
            }
            for a in result.anomalies
        ]
    )


def lab(text: str) -> pd.DataFrame:
    """Short command: Highlight invisible characters and return a code-point DataFrame."""
    print(visualize_invisible_characters(text))
    return pd.DataFrame(inspect_text_code_points(text))


def stats(result: AnalysisResult) -> None:
    """Short command: Print key dataset statistics summary."""
    s = result.stats
    print(
        f"--- KhaingLens Dataset Statistics ---\n"
        f"Analyzed Raw Records: {s.analyzed_raw_records:,}\n"
        f"Successfully Parsed: {s.successfully_parsed_records:,}\n"
        f"Malformed Records: {s.malformed_record_count:,}\n"
        f"Empty / Missing: {s.total_empty_or_missing_records:,}\n"
        f"Affected Records: {s.records_with_any_anomaly:,} ({(s.affected_record_rate or 0)*100:.1f}%)\n"
        f"Total Findings: {s.total_anomaly_findings:,} (Errors: {s.anomaly_counts_by_severity.get('error', 0)}, "
        f"Warnings: {s.anomaly_counts_by_severity.get('warning', 0)}, Info: {s.anomaly_counts_by_severity.get('info', 0)})\n"
        f"Exact Duplicate Groups: {s.duplicate_group_count:,} ({s.total_records_in_duplicate_groups:,} records affected)"
    )


def export(result: AnalysisResult, filepath: str | Path) -> Path:
    """Short command: Export analysis report to a .csv or .json file in 1 line."""
    path = Path(filepath)
    ext = path.suffix.lower()
    if ext == ".csv":
        data = generate_csv_report(result, include_utf8_bom=True)
        path.write_bytes(data)
    elif ext == ".json":
        data = generate_json_report(result)
        path.write_text(data, encoding="utf-8")
    else:
        raise ValueError("Export format must be .csv or .json")
    print(f"[KhaingLens] Report successfully exported to '{path.resolve()}'")
    return path


def launch_ui() -> None:
    """Launch the interactive KhaingLens Streamlit visual interface if app.py is present."""
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(
            "app.py was not found. Clone the full KhaingLens repository and install "
            "UI dependencies ('pip install -e .[ui]') to launch the visual dashboard."
        )
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)


__all__ = [
    "__app_name__",
    "__version__",
    "AnalysisConfig",
    "AnalysisResult",
    "Anomaly",
    "AnomalyType",
    "DatasetStats",
    "DuplicateGroup",
    "InputFormat",
    "Severity",
    "analyze_dataset",
    "export",
    "generate_csv_report",
    "generate_json_report",
    "inspect",
    "inspect_dataset_bytes",
    "inspect_text_code_points",
    "lab",
    "launch_ui",
    "load_dataset",
    "stats",
    "to_dataframe",
    "visualize_invisible_characters",
]