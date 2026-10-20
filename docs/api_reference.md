# KhaingLens v0.1.0 — Python API & Command Guide

KhaingLens provides a 1-line developer API designed for Python scripts, Jupyter notebooks, and automated data pipelines, alongside its interactive Streamlit dashboard[cite: 1].

---

## Quick Command Summary

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `kl.inspect(source)` | Inspect a file (`.csv`, `.jsonl`, `.txt`), DataFrame, or list[cite: 1] | `res = kl.inspect("data.csv")` |
| `kl.to_dataframe(res)` | Convert detected anomalies into a Pandas DataFrame[cite: 1] | `df = kl.to_dataframe(res)` |
| `kl.stats(res)` | Print dataset-level statistics and duplication summary[cite: 1] | `kl.stats(res)` |
| `kl.lab(text)` | Reveal invisible characters and return a code-point table[cite: 1] | `cp_df = kl.lab("မြန်\u200bမာစာ")` |
| `kl.export(res, path)` | Save full analysis report to `.csv` (UTF-8-BOM) or `.json`[cite: 1] | `kl.export(res, "report.csv")` |
| `kl.launch_ui()` | Launch the interactive Streamlit visual dashboard[cite: 1] | `kl.launch_ui()` |

---

## Detailed Usage Examples

### 1. Inspecting Files, DataFrames, or Lists (`kl.inspect`)
```python
import pandas as pd
import khainglens as kl

# Inspect a dataset file (.csv, .jsonl, .txt)
res = kl.inspect("sample_data/sample_burmese.csv")

# Inspect a specific column in a Pandas DataFrame
df = pd.DataFrame({"text": ["မင်္ဂလာပါ", "မင်္ဂလာပါ", "မြန်\u200bမာစာ"]})
res_df = kl.inspect(df, field="text")

# Inspect a Python list of strings
res_list = kl.inspect(["Clean text", "  Leading space", "က\u1037\u1037"])
```

### 2. Filtering Anomalies as a DataFrame (`kl.to_dataframe`)
```python
anom_df = kl.to_dataframe(res)

# View all columns: record_index, source_line_number, severity, issue_type, occurrence_count, char_position, title, evidence, original_excerpt
print(anom_df[["record_index", "severity", "issue_type", "occurrence_count"]])

# Filter only warnings and errors
warnings_and_errors = anom_df[anom_df["severity"].isin(["error", "warning"])]
```

### 3. Printing Dataset Statistics (`kl.stats`)
```python
kl.stats(res)
# Or access the underlying Pydantic model directly:
print(res.stats.min_length, res.stats.median_length, res.stats.max_length)
```

### 4. Unicode & Burmese Script Lab (`kl.lab`)
```python
# Prints visual markers for invisible characters and returns a code-point DataFrame
code_points_df = kl.lab("မင်္ဂလာပါ\u200b မြန်မာစာ")
print(code_points_df)
```

### 5. Exporting CSV & JSON Reports (`kl.export`)
```python
kl.export(res, "report.csv")   # Includes UTF-8 BOM for Microsoft Excel compatibility
kl.export(res, "report.json")  # Structured JSON report
```

### 6. Customizing Analysis Thresholds (`kl.AnalysisConfig`)
```python
cfg = kl.AnalysisConfig(
    iqr_multiplier=2.0,
    enable_myanmar_checks=True,
    max_records_limit=50000,
)
res = kl.inspect("sample_data/sample_burmese.csv", config=cfg)
```