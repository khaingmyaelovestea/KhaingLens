<p align="center">
  <img src="https://raw.githubusercontent.com/khaingmyaelovestea/KhaingLens/main/assets/logo.svg" alt="KhaingLens Logo" width="240">
</p>

# KhaingLens

**Inspect. Understand. Improve.**

KhaingLens is a local-first Python library and interactive visual toolkit designed for inspecting the quality of English and Burmese (Myanmar) text corpora. It helps machine learning engineers, NLP researchers, and dataset developers discover structural defects, whitespace anomalies, Unicode irregularities, and duplication without modifying or deleting original data.

---

## Key Features
- **Python API (`khainglens`):** Inspect `.csv`, `.jsonl`, `.txt` files, Pandas DataFrames, or Python lists directly in scripts and notebooks (see [`docs/api_reference.md`](https://github.com/khaingmyaelovestea/KhaingLens/blob/main/docs/api_reference.md)).
- **Supported Formats:** CSV, JSONL (JSON Lines), and TXT files, with explicit encoding validation (UTF-8 and UTF-8-BOM).
- **Malformed Record Isolation:** Broken CSV rows, invalid JSON lines, and decoding errors are isolated with 1-based line numbers rather than silently dropped.
- **Burmese (Myanmar) Script Awareness:** Protects valid Myanmar Unicode sequences (Kinzi, stacked consonants, medials, logical `U+1031` order) while flagging genuine structural warnings (orphaned diacritics, invisible double marks, and legacy Zawgyi-One patterns).
- **Exact & Normalized Duplication:** $O(N)$ exact raw-text duplicate grouping and Unicode NFC canonical normalization variant detection.
- **Interactive UI (Streamlit):** Light-blue WCAG AA responsive interface featuring an Overview dashboard, Anomaly Explorer with Single-Record Inspector, Unicode Lab, Statistics & Plotly charts, Reports & Exports, and Configuration settings.
- **Export Capabilities:** RFC 4180-compliant CSV reports (with UTF-8 BOM support for Microsoft Excel) and structured JSON reports.
- **Local-First Privacy:** Operates entirely in local memory with zero external API calls or telemetry.

---

## Quickstart (Python API)

```python
import khainglens as kl

res = kl.inspect("sample_data/sample_burmese.csv")  # Inspect file, DataFrame, or list
df = kl.to_dataframe(res)                           # Convert findings to DataFrame
kl.stats(res)                                       # Print dataset statistics
kl.lab("မင်္ဂလာပါ\u200b မြန်မာစာ")                     # Inspect invisible chars & code points
kl.export(res, "report.csv")                        # Export .csv or .json report
```

For full command details and configuration options, see the [**Python API & Command Guide (`docs/api_reference.md`)**](https://github.com/khaingmyaelovestea/KhaingLens/blob/main/docs/api_reference.md).

---

## Interactive Visual Dashboard

Launch the visual interface via `python -m streamlit run app.py` or `kl.launch_ui()`:

| Dataset Overview | Anomaly Explorer & Record Inspector |
| :---: | :---: |
| ![Overview Dashboard](https://raw.githubusercontent.com/khaingmyaelovestea/KhaingLens/main/assets/screenshot_overview.png) | ![Anomaly Explorer](https://raw.githubusercontent.com/khaingmyaelovestea/KhaingLens/main/assets/screenshot_explorer.png) |
| **Burmese Unicode & Invisible Character Lab** | **Dataset Statistics & Length Distributions** |
| ![Unicode Lab](https://raw.githubusercontent.com/khaingmyaelovestea/KhaingLens/main/assets/screenshot_unicode_lab.png) | ![Statistics View](https://raw.githubusercontent.com/khaingmyaelovestea/KhaingLens/main/assets/screenshot_statistics.png) |

---

## Project Structure
```text
KhaingLens/
├── app.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/config.toml
├── assets/
├── khainglens/
├── tests/
├── sample_data/
└── docs/
    ├── api_reference.md
    ├── architecture.md
    └── issue_definitions.md
```

## Acknowledgments
Special thanks to **Khant Sint Heinn** (Lead Developer at DatarrX) for critical UX testing, identifying edge cases in the Streamlit multi-page lifecycle, and providing valuable feedback on the Anomaly Explorer visual architecture and Python library design.

## Project Lead & Contact

**Khaing Myae Kyaw Swar Myo**  
*Project Lead / Student*

While KhaingLens was developed through an AI-assisted engineering workflow, this repository is managed, tested, and maintained independently by the project lead. 

If you have questions, bug reports, or suggestions for new Burmese dataset diagnostics, please feel free to reach out:
- **GitHub:** [@khaingmyaelovestea](https://github.com/khaingmyaelovestea)
- **Email:** khaingmyaekyawswarmyo25@gmail.com