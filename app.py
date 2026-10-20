"""Streamlit user interface for KhaingLens v0.1.0."""

import base64
import hashlib
import html
from collections import Counter
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from khainglens import __app_name__, __version__
from khainglens.analyzer import analyze_dataset, build_record_lookup
from khainglens.loader import inspect_file_schema, load_dataset
from khainglens.models import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisScope,
    Anomaly,
    AnomalyType,
    FileEncodingError,
    InputFormat,
    MalformedRecord,
    ParsedRecord,
    SchemaInspectionResult,
    Severity,
    UnsupportedFormatError,
)
from khainglens.reporting import (
    build_export_filename,
    generate_csv_report,
    generate_json_report,
)
from khainglens.sample_generator import generate_sample_files
from khainglens.unicode_checks import (
    inspect_text_code_points,
    visualize_invisible_characters,
)

# =====================================================================
# DEVELOPER BRANDING CONSTANTS
# =====================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
BRAND_LOGO_PATH = PROJECT_ROOT / "assets" / "khainglens_logo.svg"
BRAND_ICON_PATH = PROJECT_ROOT / "assets" / "khainglens_icon.svg"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"

PAGE_SIZE = 50

SEVERITY_META: dict[str, dict[str, str]] = {
    Severity.ERROR.value: {
        "badge": "[!] ERROR",
        "bg": "#FDE8E8",
        "border": "#F5B7B1",
        "text": "#922B21",
        "bar_color": "#D9534F",
    },
    Severity.WARNING.value: {
        "badge": "[▲] WARNING",
        "bg": "#FEF5E7",
        "border": "#FAD7A0",
        "text": "#935116",
        "bar_color": "#E67E22",
    },
    Severity.INFO.value: {
        "badge": "[i] INFO",
        "bg": "#EBF5FB",
        "border": "#A8D5F5",
        "text": "#1B4F72",
        "bar_color": "#398BD4",
    },
}

UNICODE_ISSUE_TYPES: set[AnomalyType] = {
    AnomalyType.UNEXPECTED_CONTROL_CHAR,
    AnomalyType.EMBEDDED_BOM,
    AnomalyType.ZERO_WIDTH_CHARACTER,
    AnomalyType.UNUSUAL_UNICODE_SPACE,
    AnomalyType.UNICODE_NORMALIZATION_DIFF,
    AnomalyType.NORMALIZED_DUPLICATE,
    AnomalyType.MYANMAR_ORPHAN_COMBINING,
    AnomalyType.MYANMAR_DUPLICATE_DIACRITIC,
    AnomalyType.MYANMAR_SUSPECTED_ZAWGYI,
}

SAMPLE_DATASETS: dict[str, tuple[str, str]] = {
    "sample_burmese.csv": (
        "Burmese CSV Sample (Kinzi, duplicates, whitespace, ZWSP, Zawgyi, malformed row)",
        "sample_burmese.csv",
    ),
    "sample_burmese.jsonl": (
        "Burmese JSONL Sample (NFC vs non-NFC order, embedded BOM, null/empty, broken JSON)",
        "sample_burmese.jsonl",
    ),
    "sample_english.txt": (
        "English TXT Sample (Duplicates, NBSP, ZWSP, tabs, repeated spaces, length outliers)",
        "sample_english.txt",
    ),
}


def _load_svg_data_uri(svg_path: Path) -> str | None:
    """Read an SVG or image asset and encode it as a base64 data URI for crisp inline rendering."""
    if not svg_path.exists():
        return None
    raw_bytes = svg_path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("ascii")
    ext = svg_path.suffix.lower()
    mime = "image/svg+xml" if ext == ".svg" else f"image/{ext.lstrip('.')}"
    return f"data:{mime};base64,{b64}"


def _render_html(raw_html: str) -> None:
    """Render an HTML block via st.markdown after stripping leading line indentation."""
    cleaned_lines = [line.strip() for line in raw_html.splitlines() if line.strip()]
    st.markdown(" ".join(cleaned_lines), unsafe_allow_html=True)


def inject_custom_css() -> None:
    """Inject responsive, accessible light-blue design system and Myanmar-safe typography CSS."""
    st.markdown(
        """
        <style>
        /* Global Typography & Burmese Script Line-Height Safety */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, 'Padauk', 'Myanmar Text', 'Noto Sans Myanmar', 'Pyidaungsu', sans-serif;
            color: #23354D;
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }

        /* Prevent horizontal page overflow on narrow mobile viewports (360px / 390px) */
        .stApp {
            overflow-x: hidden;
        }

        /* Sidebar Branding Block */
        .kl-sidebar-brand {
            padding: 0.25rem 0 0.75rem 0;
            border-bottom: 1px solid #DCEAF7;
            margin-bottom: 0.85rem;
        }
        .kl-sidebar-logo {
            height: 36px;
            width: auto;
            max-width: 100%;
            display: block;
        }
        .kl-sidebar-tagline {
            font-size: 0.78rem;
            color: #52667A;
            margin-top: 0.25rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }

        /* Compact Mobile Brand Header (shown only below 768px so branding stays visible when sidebar collapses) */
        .kl-mobile-brand {
            display: none;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.65rem;
            background-color: #F0F7FF;
            border: 1px solid #DCEAF7;
            border-radius: 6px;
            margin-bottom: 0.75rem;
            width: fit-content;
        }
        .kl-mobile-brand img {
            width: 20px;
            height: 20px;
        }
        .kl-mobile-brand span {
            font-size: 0.9rem;
            font-weight: 700;
            color: #23354D;
        }

        /* Page Title & Subtitle */
        .kl-page-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #23354D;
            margin: 0 0 0.2rem 0;
            line-height: 1.3;
        }
        .kl-page-subtitle {
            font-size: 0.95rem;
            color: #52667A;
            margin: 0 0 1.1rem 0;
            line-height: 1.5;
        }

        /* Burmese & Long Text Safe Containers */
        .kl-text-box {
            background-color: #F0F7FF;
            border: 1px solid #DCEAF7;
            border-radius: 6px;
            padding: 0.85rem 1rem;
            font-family: 'Padauk', 'Myanmar Text', 'Noto Sans Myanmar', 'Pyidaungsu', 'Consolas', monospace;
            font-size: 0.98rem;
            line-height: 1.85;
            color: #23354D;
            overflow-wrap: anywhere;
            word-break: break-word;
            white-space: pre-wrap;
        }

        /* Info & Panel Cards */
        .kl-panel {
            background-color: #FFFFFF;
            border: 1px solid #DCEAF7;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.9rem;
        }
        .kl-panel-subtle {
            background-color: #F0F7FF;
            border: 1px solid #DCEAF7;
            border-radius: 8px;
            padding: 0.85rem 1.1rem;
            margin-bottom: 0.9rem;
        }

        /* Accessible Severity Badges (Text + Icon + Color) */
        .kl-badge {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            white-space: nowrap;
        }
        .kl-badge-error {
            background-color: #FDE8E8;
            border: 1px solid #F5B7B1;
            color: #922B21;
        }
        .kl-badge-warning {
            background-color: #FEF5E7;
            border: 1px solid #FAD7A0;
            color: #935116;
        }
        .kl-badge-info {
            background-color: #EBF5FB;
            border: 1px solid #A8D5F5;
            color: #1B4F72;
        }

        /* Truncation Safeguard Banner */
        .kl-truncation-banner {
            background-color: #FEF5E7;
            border-left: 5px solid #E67E22;
            border-top: 1px solid #FAD7A0;
            border-right: 1px solid #FAD7A0;
            border-bottom: 1px solid #FAD7A0;
            padding: 0.85rem 1rem;
            border-radius: 6px;
            color: #935116;
            margin-bottom: 1rem;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        /* Metric Cards Responsive Styling */
        div[data-testid="stMetric"] {
            background-color: #F0F7FF;
            border: 1px solid #DCEAF7;
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
        }
        div[data-testid="stMetricLabel"] {
            color: #52667A !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: #23354D !important;
            font-size: 1.45rem !important;
        }

        /* Responsive Breakpoints (Mobile <= 768px & Narrow Mobile <= 390px) */
        @media (max-width: 768px) {
            .kl-mobile-brand {
                display: inline-flex;
            }
            .main .block-container {
                padding-left: 0.85rem;
                padding-right: 0.85rem;
                padding-top: 1rem;
            }
            .kl-page-title {
                font-size: 1.35rem;
            }
            div[data-testid="stMetricValue"] {
                font-size: 1.2rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_branding() -> None:
    """Render compact logo and tagline at the top of the left sidebar."""
    logo_uri = _load_svg_data_uri(BRAND_LOGO_PATH)
    if logo_uri:
        st.sidebar.markdown(
            f'<div class="kl-sidebar-brand">'
            f'<img src="{logo_uri}" alt="KhaingLens Logo" class="kl-sidebar-logo" />'
            f'<div class="kl-sidebar-tagline">Inspect. Understand. Improve.</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            '<div class="kl-sidebar-brand">'
            '<div style="font-size:1.35rem;font-weight:700;color:#23354D;">'
            'Khaing<span style="color:#1F6FB5;">Lens</span></div>'
            '<div class="kl-sidebar-tagline">Inspect. Understand. Improve.</div>'
            "</div>",
            unsafe_allow_html=True,
        )


def render_page_header(title: str, subtitle: str) -> None:
    """Render mobile compact branding pill (visible only <=768px) and main page title/subtitle."""
    icon_uri = _load_svg_data_uri(BRAND_ICON_PATH)
    icon_html = f'<img src="{icon_uri}" alt="KhaingLens Icon" />' if icon_uri else ""
    _render_html(
        f"""
        <div class="kl-mobile-brand">
            {icon_html}
            <span>KhaingLens</span>
        </div>
        <h1 class="kl-page-title">{html.escape(title)}</h1>
        <p class="kl-page-subtitle">{html.escape(subtitle)}</p>
        """
    )


def render_severity_badge(severity: Severity | str) -> str:
    """Return an accessible HTML badge with symbol + text label for a severity level."""
    sev_val = severity.value if isinstance(severity, Severity) else str(severity)
    meta = SEVERITY_META.get(sev_val, SEVERITY_META[Severity.INFO.value])
    css_class = f"kl-badge kl-badge-{sev_val}"
    return f'<span class="{css_class}">{html.escape(meta["badge"])}</span>'


def render_truncation_banner_if_needed(result: AnalysisResult) -> None:
    """Display a prominent warning banner whenever the dataset exceeded max_records_limit."""
    lr = result.load_result
    if lr.truncated_at_limit or lr.analysis_scope == AnalysisScope.TRUNCATED_SUBSET:
        _render_html(
            f"""
            <div class="kl-truncation-banner">
                <strong>[▲] Partial Dataset Analysis (Truncated at Limit):</strong>
                This dataset exceeded the configured maximum limit of
                <strong>{lr.max_records_limit_applied:,}</strong> records. KhaingLens analyzed only the
                <strong>first {lr.analyzed_raw_records:,} records</strong>
                (<code>analysis_scope = truncated_subset</code>). All statistics, duplicate groups,
                outlier fences, and exported reports apply strictly to this analyzed subset and must
                not be interpreted as complete-dataset results. You can raise the limit up to 500,000
                in <strong>Analysis Configuration</strong>.
            </div>
            """
        )


@st.cache_data(show_spinner=False, max_entries=5)
def cached_inspect_schema(
    file_bytes: bytes,
    filename: str,
    explicit_format_str: str | None,
    csv_delimiter: str | None,
    csv_has_header: bool,
) -> SchemaInspectionResult:
    """Cached wrapper around inspect_file_schema."""
    fmt = InputFormat(explicit_format_str) if explicit_format_str else None
    return inspect_file_schema(
        file_bytes=file_bytes,
        filename=filename,
        explicit_format=fmt,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
    )


@st.cache_data(show_spinner=False, max_entries=3)
def cached_load_and_analyze(
    file_hash: str,
    file_bytes: bytes,
    filename: str,
    config_json: str,
    explicit_format_str: str | None,
    selected_field: str | None,
    csv_delimiter: str | None,
    csv_has_header: bool,
) -> AnalysisResult:
    """Cached wrapper around load_dataset and analyze_dataset keyed on file hash and settings."""
    _ = file_hash  # Included in cache key
    config = AnalysisConfig.model_validate_json(config_json)
    fmt = InputFormat(explicit_format_str) if explicit_format_str else None
    load_res = load_dataset(
        file_bytes=file_bytes,
        filename=filename,
        config=config,
        explicit_format=fmt,
        selected_field=selected_field,
        csv_delimiter=csv_delimiter,
        csv_has_header=csv_has_header,
    )
    return analyze_dataset(load_result=load_res, config=config)


def init_session_state() -> None:
    """Initialize Streamlit session state defaults with persistent uploaded dataset storage."""
    if "analysis_config" not in st.session_state:
        st.session_state["analysis_config"] = AnalysisConfig()
    if "active_sample_name" not in st.session_state:
        st.session_state["active_sample_name"] = None
    if "persisted_file_bytes" not in st.session_state:
        st.session_state["persisted_file_bytes"] = None
    if "persisted_file_name" not in st.session_state:
        st.session_state["persisted_file_name"] = None
    if "csv_has_header" not in st.session_state:
        st.session_state["csv_has_header"] = True
    if "csv_delimiter_choice" not in st.session_state:
        st.session_state["csv_delimiter_choice"] = "Auto-detect"
    if "selected_field_override" not in st.session_state:
        st.session_state["selected_field_override"] = None
    if "inspector_record_index" not in st.session_state:
        st.session_state["inspector_record_index"] = 0


def _ensure_samples_exist() -> None:
    """Ensure synthetic sample files exist on disk before loading."""
    if not (SAMPLE_DATA_DIR / "sample_burmese.csv").exists():
        generate_sample_files(SAMPLE_DATA_DIR)


def _get_active_file_payload() -> tuple[bytes | None, str | None, str]:
    """Return (file_bytes, filename, source_type) persisted across all page switches."""
    # 1. If the user just uploaded a file via the widget on Overview, persist it immediately
    uploaded = st.session_state.get("uploaded_dataset_widget")
    if uploaded is not None:
        new_bytes = uploaded.getvalue()
        new_name = uploaded.name
        if (
            st.session_state.get("persisted_file_name") != new_name
            or st.session_state.get("persisted_file_bytes") != new_bytes
        ):
            st.session_state["persisted_file_bytes"] = new_bytes
            st.session_state["persisted_file_name"] = new_name
            st.session_state["active_sample_name"] = None
            st.session_state["selected_field_override"] = None

    # 2. If a bundled sample dataset is explicitly selected, use the sample
    sample_name = st.session_state.get("active_sample_name")
    if sample_name and sample_name in SAMPLE_DATASETS:
        _ensure_samples_exist()
        sample_path = SAMPLE_DATA_DIR / SAMPLE_DATASETS[sample_name][1]
        if sample_path.exists():
            return sample_path.read_bytes(), sample_path.name, "sample"

    # 3. Otherwise, return the persisted uploaded file (survives all tab switches!)
    if (
        st.session_state.get("persisted_file_bytes") is not None
        and st.session_state.get("persisted_file_name") is not None
    ):
        return (
            st.session_state["persisted_file_bytes"],
            st.session_state["persisted_file_name"],
            "uploaded",
        )

    return None, None, "none"


# =====================================================================
# VIEW 1: OVERVIEW
# =====================================================================
def render_overview_page(
    result: AnalysisResult | None,
    schema_info: SchemaInspectionResult | None,
    active_filename: str | None,
    load_error: str | None,
) -> None:
    """Render the Overview & Dataset Upload section."""
    render_page_header(
        "Dataset Overview & Inspection Setup",
        "Upload a CSV, JSONL, or TXT dataset (or load a synthetic sample) to inspect data quality without modifying your source records.",
    )

    with st.container():
        col_up, col_sample = st.columns([3, 2])
        with col_up:
            uploaded_file = st.file_uploader(
                "Upload Dataset (.csv, .jsonl, .txt)",
                type=["csv", "jsonl", "ndjson", "txt"],
                key="uploaded_dataset_widget",
                help="Processed locally in memory. Files are never modified or sent to external AI services.",
            )
            if uploaded_file is not None:
                st.session_state["persisted_file_bytes"] = uploaded_file.getvalue()
                st.session_state["persisted_file_name"] = uploaded_file.name
                if st.session_state.get("active_sample_name") is not None:
                    st.session_state["active_sample_name"] = None
                    st.session_state["selected_field_override"] = None
                    st.rerun()

            if (
                st.session_state.get("persisted_file_name") is not None
                and st.session_state.get("active_sample_name") is None
            ):
                c_info, c_btn = st.columns([3, 1])
                with c_info:
                    st.success(
                        f"Active Uploaded File: **`{st.session_state['persisted_file_name']}`**"
                    )
                with c_btn:
                    if st.button("Clear File", use_container_width=True):
                        st.session_state["persisted_file_bytes"] = None
                        st.session_state["persisted_file_name"] = None
                        st.session_state["selected_field_override"] = None
                        st.rerun()

        with col_sample:
            st.markdown("**Or Try a Synthetic Sample Dataset**")
            sample_options = ["None (Use Uploaded File)"] + list(SAMPLE_DATASETS.keys())
            current_sample = st.session_state.get("active_sample_name")
            default_idx = (
                sample_options.index(current_sample)
                if current_sample in sample_options
                else 0
            )
            chosen_sample = st.selectbox(
                "Bundled Sample Datasets",
                options=sample_options,
                index=default_idx,
                format_func=lambda x: SAMPLE_DATASETS[x][0] if x in SAMPLE_DATASETS else x,
                help="Synthetic English and Burmese samples demonstrating duplicates, whitespace issues, Unicode anomalies, and malformed rows.",
            )
            if chosen_sample == "None (Use Uploaded File)":
                if st.session_state.get("active_sample_name") is not None:
                    st.session_state["active_sample_name"] = None
                    st.session_state["selected_field_override"] = None
                    st.rerun()
            else:
                if st.session_state.get("active_sample_name") != chosen_sample:
                    st.session_state["active_sample_name"] = chosen_sample
                    st.session_state["selected_field_override"] = None
                    st.rerun()

    if load_error:
        st.error(f"**Dataset Load Error:** {load_error}")
        return

    if result is None or schema_info is None or active_filename is None:
        _render_html(
            """
            <div class="kl-panel-subtle">
                <strong>Ready to Inspect:</strong> Upload a <code>.csv</code>, <code>.jsonl</code>, or
                <code>.txt</code> file above, or select one of the bundled Burmese/English sample datasets
                on the right to begin inspection.
            </div>
            """
        )
        return

    # Parsing & Field Selection Controls
    lr = result.load_result
    st.markdown("### Input Format & Parsing Controls")
    ctrl_cols = st.columns(4)

    with ctrl_cols[0]:
        st.markdown(
            f"**Detected Format:** `{lr.file_format.value.upper()}`  \n"
            f"**Encoding:** `{lr.encoding_used}` {'(BOM stripped)' if lr.has_bom else ''}"
        )

    with ctrl_cols[1]:
        if lr.file_format == InputFormat.CSV:
            new_has_header = st.checkbox(
                "File has a header row",
                value=st.session_state["csv_has_header"],
                help="When unchecked, the first row is analyzed as data (record_index=0) and columns are named col_0, col_1, etc.",
            )
            if new_has_header != st.session_state["csv_has_header"]:
                st.session_state["csv_has_header"] = new_has_header
                st.session_state["selected_field_override"] = None
                st.rerun()
        else:
            st.caption("Header toggle applies to CSV files.")

    with ctrl_cols[2]:
        if lr.file_format == InputFormat.CSV:
            delim_options = ["Auto-detect", "Comma (,)", "Tab (\\t)", "Semicolon (;)", "Pipe (|)"]
            chosen_delim = st.selectbox(
                "CSV Delimiter",
                options=delim_options,
                index=delim_options.index(st.session_state["csv_delimiter_choice"]),
            )
            if chosen_delim != st.session_state["csv_delimiter_choice"]:
                st.session_state["csv_delimiter_choice"] = chosen_delim
                st.rerun()
        else:
            st.caption("Line-delimited record format.")

    with ctrl_cols[3]:
        if lr.file_format in (InputFormat.CSV, InputFormat.JSONL) and schema_info.candidate_fields:
            candidates = schema_info.candidate_fields
            active_field = lr.selected_field if lr.selected_field in candidates else candidates[0]
            chosen_field = st.selectbox(
                "Target Text Column / Field",
                options=candidates,
                index=candidates.index(active_field),
                help="Select which column (CSV) or key (JSONL) contains the text to inspect.",
            )
            if chosen_field != lr.selected_field:
                st.session_state["selected_field_override"] = chosen_field
                st.rerun()
        else:
            st.markdown(f"**Target Field:** `{lr.selected_field or 'text'}`")

    render_truncation_banner_if_needed(result)

    for warn_msg in lr.load_warnings:
        st.warning(warn_msg)

    # Top-Level Summary Metrics
    st.markdown("### Inspection Summary")
    stats = result.stats
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Analyzed Raw Records", f"{stats.analyzed_raw_records:,}")
    m2.metric("Analyzable Non-Empty", f"{stats.non_empty_analyzable_records:,}")
    m3.metric("Malformed Records", f"{stats.malformed_record_count:,}")
    m4.metric("Empty / Missing", f"{stats.total_empty_or_missing_records:,}")
    m5.metric(
        "Affected Records",
        f"{stats.records_with_any_anomaly:,}",
        delta=(
            f"{stats.affected_record_rate * 100:.1f}% of dataset"
            if stats.affected_record_rate is not None
            else None
        ),
        delta_color="off",
    )

    # Severity & Top Issue Breakdown
    st.markdown("### Detected Findings Breakdown")
    st.caption(
        f"**Note on Counting:** KhaingLens detected **{stats.total_anomaly_findings:,} consolidated anomaly findings** "
        f"(representing **{stats.total_in_record_anomaly_occurrences:,} total in-record occurrences**) across "
        f"**{stats.records_with_any_anomaly:,} unique affected records**."
    )

    sev_col1, sev_col2, sev_col3 = st.columns(3)
    for col_ui, sev_enum in zip(
        (sev_col1, sev_col2, sev_col3),
        (Severity.ERROR, Severity.WARNING, Severity.INFO),
    ):
        s_key = sev_enum.value
        finding_cnt = stats.anomaly_counts_by_severity.get(s_key, 0)
        rec_cnt = stats.affected_records_by_severity.get(s_key, 0)
        with col_ui:
            _render_html(
                f"""
                <div class="kl-panel">
                    {render_severity_badge(sev_enum)}
                    <div style="font-size:1.35rem;font-weight:700;margin-top:0.4rem;">
                        {finding_cnt:,} finding(s)
                    </div>
                    <div style="font-size:0.86rem;color:#52667A;">
                        Affecting <strong>{rec_cnt:,}</strong> unique record(s)
                    </div>
                </div>
                """
            )

    if stats.anomaly_counts_by_type:
        type_to_sev = {a.issue_type.value: a.severity.value for a in result.anomalies}
        sorted_issues = sorted(
            stats.anomaly_counts_by_type.items(), key=lambda x: (x[1], x[0])
        )
        y_types = [k for k, _ in sorted_issues]
        x_recs = [v for _, v in sorted_issues]
        bar_colors = [
            SEVERITY_META[type_to_sev.get(k, Severity.INFO.value)]["bar_color"]
            for k in y_types
        ]

        fig_overview = go.Figure(
            data=[
                go.Bar(
                    x=x_recs,
                    y=y_types,
                    orientation="h",
                    marker_color=bar_colors,
                    text=x_recs,
                    textposition="auto",
                    hovertemplate="Issue Type: %{y}<br>Affected Records: %{x:,}<extra></extra>",
                )
            ]
        )
        fig_overview.update_layout(
            title=dict(
                text="Affected Records by Issue Type (Color-Coded by Severity)",
                font=dict(size=14, color="#23354D"),
            ),
            xaxis_title="Affected Records",
            yaxis_title="",
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#23354D", size=12),
            margin=dict(l=16, r=24, t=40, b=36),
            height=max(260, len(y_types) * 32 + 80),
        )
        st.plotly_chart(fig_overview, use_container_width=True, config={"displayModeBar": False})

        with st.expander("View Issue Summary as Table", expanded=False):
            rows = []
            for issue_key, count in sorted(
                stats.anomaly_counts_by_type.items(), key=lambda x: (-x[1], x[0])
            ):
                sev_str = type_to_sev.get(issue_key, Severity.INFO.value)
                rows.append(
                    {
                        "Issue Type": issue_key,
                        "Severity": SEVERITY_META[sev_str]["badge"],
                        "Affected Records (Findings)": count,
                        "Total In-Record Occurrences": stats.occurrence_sums_by_type.get(issue_key, count),
                        "Share of Analyzed Records": (
                            f"{(count / stats.analyzed_raw_records) * 100:.1f}%"
                            if stats.analyzed_raw_records > 0
                            else "N/A"
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("No quality anomalies were detected in the analyzed dataset!")


# =====================================================================
# VIEW 2: ANOMALY EXPLORER (VISUAL-FIRST INSIGHTS + RECORD INSPECTOR)
# =====================================================================
def render_anomaly_explorer_page(result: AnalysisResult | None) -> None:
    """Render the visual-first Anomaly Explorer and Single-Record Inspector."""
    render_page_header(
        "Anomaly Explorer",
        "Explore visual breakdowns of detected data quality issues, filter by category, and inspect individual problematic records.",
    )
    if result is None:
        st.info("Please upload a dataset or select a sample dataset on the **Overview** page first.")
        return

    render_truncation_banner_if_needed(result)
    stats = result.stats

    # Filter Bar
    f_col1, f_col2, f_col3 = st.columns([2, 3, 2])
    with f_col1:
        selected_severities = st.multiselect(
            "Filter by Severity",
            options=[s.value for s in Severity],
            default=[s.value for s in Severity],
            format_func=lambda s: SEVERITY_META[s]["badge"],
        )
    with f_col2:
        present_types = sorted({a.issue_type.value for a in result.anomalies})
        selected_types = st.multiselect(
            "Filter by Anomaly Type",
            options=present_types,
            default=present_types,
        )
    with f_col3:
        search_query = st.text_input(
            "Search Text Excerpt or Record Index",
            placeholder="e.g., မြန်မာ or 7",
        ).strip()

    filtered: list[Anomaly] = []
    for anom in result.anomalies:
        if anom.severity.value not in selected_severities:
            continue
        if anom.issue_type.value not in selected_types:
            continue
        if search_query:
            q_lower = search_query.lower()
            matches_idx = search_query.isdigit() and int(search_query) == anom.record_index
            matches_text = (
                q_lower in anom.original_excerpt.lower()
                or q_lower in anom.title.lower()
                or q_lower in anom.evidence.lower()
            )
            if not (matches_idx or matches_text):
                continue
        filtered.append(anom)

    filtered_unique_records = len({a.record_index for a in filtered})
    filtered_total_occurrences = sum(a.occurrence_count for a in filtered)

    # Filtered KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Matching Findings", f"{len(filtered):,}")
    k2.metric("Affected Records", f"{filtered_unique_records:,}")
    k3.metric("Total In-Record Occurrences", f"{filtered_total_occurrences:,}")
    k4.metric(
        "Share of Dataset Affected",
        (
            f"{(filtered_unique_records / stats.analyzed_raw_records) * 100:.1f}%"
            if stats.analyzed_raw_records > 0
            else "0.0%"
        ),
    )

    if not filtered:
        st.info("No anomalies match your current filter criteria.")
    else:
        # VISUAL INSIGHTS FIRST (Clean headers + non-overlapping legend)
        st.markdown("### Visual Breakdown of Filtered Findings")
        v_col1, v_col2 = st.columns([2, 3])

        with v_col1:
            st.markdown("#### Findings by Severity Level")
            sev_counts = Counter(a.severity.value for a in filtered)
            sev_order = [
                s.value for s in (Severity.ERROR, Severity.WARNING, Severity.INFO)
                if sev_counts.get(s.value, 0) > 0
            ]
            sev_labels = [SEVERITY_META[s]["badge"] for s in sev_order]
            sev_values = [sev_counts[s] for s in sev_order]
            sev_colors = [SEVERITY_META[s]["bar_color"] for s in sev_order]

            fig_sev = go.Figure(
                data=[
                    go.Pie(
                        labels=sev_labels,
                        values=sev_values,
                        hole=0.52,
                        marker=dict(colors=sev_colors, line=dict(color="#FFFFFF", width=2)),
                        textinfo="label+percent",
                        hovertemplate="%{label}<br>Findings: %{value:,} (%{percent})<extra></extra>",
                    )
                ]
            )
            fig_sev.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#23354D", size=12),
                margin=dict(l=12, r=12, t=16, b=20),
                height=320,
                showlegend=False,
            )
            st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})

        with v_col2:
            st.markdown("#### Issue Frequency (Affected Records vs. Total Occurrences)")
            type_finding_counts = Counter(a.issue_type.value for a in filtered)
            type_occ_counts: dict[str, int] = {}
            type_sev_lookup: dict[str, str] = {}
            for a in filtered:
                k = a.issue_type.value
                type_occ_counts[k] = type_occ_counts.get(k, 0) + a.occurrence_count
                type_sev_lookup[k] = a.severity.value

            sorted_filtered_types = sorted(
                type_finding_counts.items(), key=lambda x: (x[1], x[0])
            )
            y_labels = [k for k, _ in sorted_filtered_types]
            x_findings = [v for _, v in sorted_filtered_types]
            x_occurrences = [type_occ_counts[k] for k in y_labels]
            colors_bar = [
                SEVERITY_META[type_sev_lookup.get(k, Severity.INFO.value)]["bar_color"]
                for k in y_labels
            ]

            fig_types = go.Figure()
            fig_types.add_trace(
                go.Bar(
                    name="Affected Records (Findings)",
                    x=x_findings,
                    y=y_labels,
                    orientation="h",
                    marker_color=colors_bar,
                    text=x_findings,
                    textposition="auto",
                    hovertemplate="Issue: %{y}<br>Affected Records: %{x:,}<extra></extra>",
                )
            )
            fig_types.add_trace(
                go.Bar(
                    name="Total In-Record Occurrences",
                    x=x_occurrences,
                    y=y_labels,
                    orientation="h",
                    marker_color="#A8D5F5",
                    hovertemplate="Issue: %{y}<br>Total Occurrences: %{x:,}<extra></extra>",
                )
            )
            fig_types.update_layout(
                barmode="group",
                xaxis_title="Count",
                yaxis=dict(title="", automargin=True),
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#23354D", size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                ),
                margin=dict(l=12, r=20, t=36, b=30),
                height=max(320, len(y_labels) * 36 + 80),
            )
            st.plotly_chart(fig_types, use_container_width=True, config={"displayModeBar": False})

        # Optional Raw Table tucked neatly inside an Expander
        with st.expander(f"View Filtered Findings as Raw Table ({len(filtered):,} rows)", expanded=False):
            total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
            p_col1, _ = st.columns([1, 3])
            with p_col1:
                page_num = st.number_input(
                    f"Page (1 to {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                )
            start_idx = (page_num - 1) * PAGE_SIZE
            page_items = filtered[start_idx : start_idx + PAGE_SIZE]

            table_rows = [
                {
                    "Record Index": a.record_index,
                    "Source Line": a.source_line_number or "",
                    "Severity": SEVERITY_META[a.severity.value]["badge"],
                    "Issue Type": a.issue_type.value,
                    "Occurrences": a.occurrence_count,
                    "First Char Pos": a.char_position if a.char_position is not None else "",
                    "Title": a.title,
                    "Excerpt": a.original_excerpt[:120],
                }
                for a in page_items
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # Single-Record Deep Inspector with Smart "Jump to Affected Record" Selector
    st.markdown("---")
    st.markdown("### Individual Record Inspector")
    max_rec_idx = max(0, stats.analyzed_raw_records - 1)
    if stats.analyzed_raw_records == 0:
        st.caption("No records available to inspect.")
        return

    lookup = build_record_lookup(result.load_result)

    # Build smart list of affected record indices from current filter
    affected_indices_in_filter = sorted({a.record_index for a in filtered})
    anoms_by_rec: dict[int, list[Anomaly]] = {}
    for a in result.anomalies:
        anoms_by_rec.setdefault(a.record_index, []).append(a)

    nav_col1, nav_col2 = st.columns([3, 1])
    with nav_col1:
        if affected_indices_in_filter:
            def _format_record_option(r_idx: int) -> str:
                r_anoms = anoms_by_rec.get(r_idx, [])
                issue_names = ", ".join(a.issue_type.value for a in r_anoms[:2])
                if len(r_anoms) > 2:
                    issue_names += f" (+{len(r_anoms) - 2} more)"
                r_obj = lookup.get(r_idx)
                line_no = r_obj.source_line_number if r_obj else "?"
                preview = ""
                if isinstance(r_obj, ParsedRecord) and r_obj.raw_text:
                    preview = r_obj.raw_text[:45].replace("\n", " ")
                elif isinstance(r_obj, MalformedRecord):
                    preview = r_obj.raw_excerpt[:45].replace("\n", " ")
                return f"Record #{r_idx} (Line {line_no}) — [{len(r_anoms)} issue(s): {issue_names}] — {preview}"

            current_idx = st.session_state.get("inspector_record_index", affected_indices_in_filter[0])
            select_default_idx = (
                affected_indices_in_filter.index(current_idx)
                if current_idx in affected_indices_in_filter
                else 0
            )
            chosen_affected_idx = st.selectbox(
                f"Quick-Jump to Filtered Problematic Record ({len(affected_indices_in_filter):,} matching record(s))",
                options=affected_indices_in_filter,
                index=select_default_idx,
                format_func=_format_record_option,
                help="Select any problematic record directly without hunting through a table.",
            )
            st.session_state["inspector_record_index"] = int(chosen_affected_idx)

    with nav_col2:
        default_rec = min(st.session_state.get("inspector_record_index", 0), max_rec_idx)
        inspected_idx = st.number_input(
            f"Or Any Record Index (0–{max_rec_idx})",
            min_value=0,
            max_value=max_rec_idx,
            value=default_rec,
            step=1,
        )
        st.session_state["inspector_record_index"] = int(inspected_idx)

    active_rec_idx = int(st.session_state["inspector_record_index"])
    rec_obj = lookup.get(active_rec_idx)
    rec_anomalies = anoms_by_rec.get(active_rec_idx, [])

    if rec_obj is None:
        st.warning(f"Record #{active_rec_idx} not found in lookup.")
        return

    if isinstance(rec_obj, MalformedRecord):
        _render_html(
            f"<p><strong>Record #{rec_obj.record_index}</strong> | "
            f"<strong>Source Line:</strong> <code>{rec_obj.source_line_number}</code> | "
            f"<strong>Status:</strong> {render_severity_badge(Severity.ERROR)} Malformed Record</p>"
        )
        st.markdown("**Raw Unparsed Excerpt:**")
        st.markdown(
            f'<div class="kl-text-box">{html.escape(rec_obj.raw_excerpt)}</div>',
            unsafe_allow_html=True,
        )
    else:
        raw_display = "<MISSING / NULL>" if rec_obj.raw_text is None else rec_obj.raw_text
        char_len = len(rec_obj.raw_text) if rec_obj.raw_text is not None else 0
        st.markdown(
            f"**Record #{rec_obj.record_index}** | **Source Line:** `{rec_obj.source_line_number}` | "
            f"**Length:** `{char_len:,}` Unicode code points | "
            f"**Findings on this record:** `{len(rec_anomalies)}`",
        )
        st.markdown("**Original Unmodified Text:**")
        st.markdown(
            f'<div class="kl-text-box">{html.escape(raw_display)}</div>',
            unsafe_allow_html=True,
        )
        if rec_obj.extra_fields_preview:
            st.caption(f"Other fields in record: `{rec_obj.extra_fields_preview}`")

    if not rec_anomalies:
        st.success("No anomalies flagged on this record.")
    else:
        for anom in rec_anomalies:
            positions_str = (
                f" | Positions: {html.escape(str(anom.char_positions[:10]))}"
                if anom.char_positions
                else ""
            )
            _render_html(
                f"""
                <div class="kl-panel">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
                        <div>
                            {render_severity_badge(anom.severity)}
                            <strong style="margin-left:0.5rem;font-size:1.02rem;">{html.escape(anom.title)}</strong>
                            <code style="margin-left:0.4rem;">{html.escape(anom.issue_type.value)}</code>
                        </div>
                        <div style="font-size:0.85rem;color:#52667A;">
                            Occurrences in record: <strong>{anom.occurrence_count}</strong>{positions_str}
                        </div>
                    </div>
                    <p style="margin:0.55rem 0 0.35rem 0;font-size:0.93rem;">{html.escape(anom.explanation)}</p>
                    <div style="font-size:0.88rem;color:#23354D;background:#F0F7FF;padding:0.45rem 0.65rem;border-radius:4px;margin-bottom:0.35rem;">
                        <strong>Evidence:</strong> {html.escape(anom.evidence)}
                    </div>
                    <div style="font-size:0.88rem;color:#1F6FB5;">
                        <strong>Suggested Action:</strong> {html.escape(anom.suggested_action)}
                    </div>
                </div>
                """
            )


# =====================================================================
# VIEW 3: UNICODE LAB
# =====================================================================
def render_unicode_lab_page(result: AnalysisResult | None) -> None:
    """Render the Unicode Diagnostics & Burmese Script Lab."""
    render_page_header(
        "Unicode & Burmese Script Lab",
        "Inspect invisible characters, Unicode normalization differences, and Burmese (Myanmar) script sequences.",
    )

    with st.expander("Burmese (Myanmar) Unicode Reference & Diagnostic Policy", expanded=False):
        st.markdown(
            """
            - **Valid Myanmar Combining Sequences Are Protected:** Myanmar script (`U+1000`–`U+109F`) uses combining marks (`Mn`/`Mc`) for dependent vowels, medials (`U+103B`–`U+103E`), virama/asat (`U+1039`–`U+103A`), and tone marks (`U+1036`–`U+1038`). Sequences such as **Kinzi** (`င်္` = `U+1004 U+103A U+1039` + Consonant), stacked consonants, and logical-order pre-base vowel `ေ` (`Consonant + [Medial] + U+1031`) are valid Unicode and are **never** flagged as errors.
            - **Flagged Myanmar Warnings (`[▲] WARNING`):**
              1. `myanmar_orphan_combining`: A combining mark appearing at the start of text or after whitespace/punctuation without a base consonant.
              2. `myanmar_duplicate_diacritic`: Adjacent identical vowels, medials, or tone marks (e.g., `U+1037 U+1037`) that stack invisibly on screen.
              3. `myanmar_suspected_zawgyi`: Heuristic indicators of legacy Zawgyi-One encoding (such as visual-order `U+1031` before a consonant, trailing `U+1039`, or code points `U+1033`/`U+1034`/`U+105A`). Note that `U+1033`/`U+1034`/`U+105A` are valid in Mon ethnic script corpora.
            - **NFC Normalization Limitation:** Standard Unicode `NFC` reorders marks with distinct non-zero Canonical Combining Classes (e.g., `U+1037` dot below `CCC=7` vs. `U+103A` asat `CCC=9`), but does not reorder marks that share `CCC=0`.
            """
        )

    if result is not None:
        render_truncation_banner_if_needed(result)
        unicode_anomalies = [
            a for a in result.anomalies if a.issue_type in UNICODE_ISSUE_TYPES
        ]
        st.markdown(
            f"### Dataset Unicode Findings ({len(unicode_anomalies):,} finding(s))"
        )
        if unicode_anomalies:
            uni_counts = Counter(a.issue_type.value for a in unicode_anomalies)
            uni_sev = {a.issue_type.value: a.severity.value for a in unicode_anomalies}
            sorted_uni = sorted(uni_counts.items(), key=lambda x: (x[1], x[0]))
            u_labels = [k for k, _ in sorted_uni]
            u_vals = [v for _, v in sorted_uni]
            u_colors = [
                SEVERITY_META[uni_sev.get(k, Severity.WARNING.value)]["bar_color"]
                for k in u_labels
            ]

            fig_uni = go.Figure(
                data=[
                    go.Bar(
                        x=u_vals,
                        y=u_labels,
                        orientation="h",
                        marker_color=u_colors,
                        text=u_vals,
                        textposition="auto",
                        hovertemplate="Unicode Issue: %{y}<br>Affected Records: %{x:,}<extra></extra>",
                    )
                ]
            )
            fig_uni.update_layout(
                xaxis_title="Affected Records",
                yaxis_title="",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#23354D", size=12),
                margin=dict(l=12, r=16, t=16, b=30),
                height=max(200, len(u_labels) * 38 + 60),
            )
            st.plotly_chart(fig_uni, use_container_width=True, config={"displayModeBar": False})

            with st.expander("View Unicode Findings Table", expanded=False):
                uni_rows = [
                    {
                        "Record Index": a.record_index,
                        "Line": a.source_line_number or "",
                        "Severity": SEVERITY_META[a.severity.value]["badge"],
                        "Issue Type": a.issue_type.value,
                        "Occurrences": a.occurrence_count,
                        "First Pos": a.char_position if a.char_position is not None else "",
                        "Evidence": a.evidence,
                    }
                    for a in unicode_anomalies[:200]
                ]
                st.dataframe(pd.DataFrame(uni_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No Unicode or Burmese script anomalies detected in the current dataset.")

    st.markdown("---")
    st.markdown("### Interactive Code-Point & Invisible Character Inspector")

    inspect_mode = st.radio(
        "Inspection Source",
        options=["Select Record from Loaded Dataset", "Paste Custom Text Snippet"],
        horizontal=True,
    )

    sample_text = ""
    if inspect_mode == "Select Record from Loaded Dataset":
        if result is None or result.stats.analyzed_raw_records == 0:
            st.info("Load a dataset on the Overview page, or switch to 'Paste Custom Text Snippet'.")
            return
        lookup = build_record_lookup(result.load_result)
        max_idx = result.stats.analyzed_raw_records - 1
        rec_idx = st.number_input(
            f"Record Index (0 to {max_idx})",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1,
            key="unicode_lab_rec_idx",
        )
        rec_obj = lookup.get(int(rec_idx))
        if isinstance(rec_obj, ParsedRecord) and rec_obj.raw_text is not None:
            sample_text = rec_obj.raw_text
        elif isinstance(rec_obj, MalformedRecord):
            sample_text = rec_obj.raw_excerpt
    else:
        sample_text = st.text_area(
            "Enter or paste English/Burmese text to inspect code points",
            value="မင်္ဂလာပါ\u200b မြန်မာစာ က\u103a\u1037",
            height=100,
        )

    if sample_text:
        st.markdown("**Visualized Stream (Invisible & Special Characters Highlighted):**")
        vis_str = visualize_invisible_characters(sample_text)
        st.markdown(
            f'<div class="kl-text-box">{html.escape(vis_str)}</div>',
            unsafe_allow_html=True,
        )

        cp_rows = inspect_text_code_points(sample_text, max_chars=300)
        with st.expander(
            f"View Character-by-Character Code Point Table ({len(cp_rows)} of {len(sample_text)} code points)",
            expanded=True,
        ):
            st.dataframe(pd.DataFrame(cp_rows), use_container_width=True, hide_index=True)


# =====================================================================
# VIEW 4: STATISTICS
# =====================================================================
def render_statistics_page(result: AnalysisResult | None) -> None:
    """Render comprehensive dataset statistics and responsive Plotly charts."""
    render_page_header(
        "Dataset Statistics & Distributions",
        "Quantitative breakdown of record counts, duplication rates, and text length distributions.",
    )
    if result is None:
        st.info("Please upload a dataset or select a sample dataset on the **Overview** page first.")
        return

    render_truncation_banner_if_needed(result)
    stats = result.stats

    # Section 1: Record & Blank Breakdown
    st.markdown("### Record & Missingness Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analyzed Raw Records", f"{stats.analyzed_raw_records:,}")
    c2.metric("Successfully Parsed", f"{stats.successfully_parsed_records:,}")
    c3.metric(
        "Malformed Records",
        f"{stats.malformed_record_count:,}",
        delta=(
            f"{stats.malformed_record_rate * 100:.2f}% of raw"
            if stats.malformed_record_rate is not None
            else None
        ),
        delta_color="off",
    )
    c4.metric(
        "Empty / Missing Total",
        f"{stats.total_empty_or_missing_records:,}",
        delta=(
            f"{stats.empty_or_missing_rate * 100:.2f}% of parsed"
            if stats.empty_or_missing_rate is not None
            else None
        ),
        delta_color="off",
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Missing / Null Values", f"{stats.missing_value_record_count:,}")
    b2.metric("Empty Strings ('')", f"{stats.empty_string_record_count:,}")
    b3.metric("Whitespace-Only", f"{stats.whitespace_only_record_count:,}")
    b4.metric("Non-Empty Analyzable", f"{stats.non_empty_analyzable_records:,}")

    # Section 2: Duplication Metrics
    st.markdown("### Duplication Metrics (Over Non-Empty Analyzable Records)")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Unique Non-Empty Texts", f"{stats.unique_non_empty_texts:,}")
    d2.metric("Exact Duplicate Groups", f"{stats.duplicate_group_count:,}")
    d3.metric(
        "Records in Dup Groups",
        f"{stats.total_records_in_duplicate_groups:,}",
        delta=(
            f"{stats.duplicate_involvement_rate * 100:.1f}% of analyzable"
            if stats.duplicate_involvement_rate is not None
            else None
        ),
        delta_color="off",
    )
    d4.metric(
        "Surplus Dup Occurrences",
        f"{stats.surplus_duplicate_occurrences:,}",
        delta=(
            f"{stats.surplus_duplicate_rate * 100:.1f}% redundant"
            if stats.surplus_duplicate_rate is not None
            else None
        ),
        delta_color="off",
    )

    if result.duplicate_groups:
        with st.expander(f"View Exact Duplicate Groups ({len(result.duplicate_groups)})"):
            dup_df = pd.DataFrame(
                [
                    {
                        "Group ID": g.group_id,
                        "Occurrences": g.occurrence_count,
                        "Surplus Copies": g.surplus_count,
                        "Record Indices": str(g.record_indices[:20]),
                        "Excerpt": g.representative_excerpt[:120],
                    }
                    for g in result.duplicate_groups
                ]
            )
            st.dataframe(dup_df, use_container_width=True, hide_index=True)

    # Section 3: Length Statistics (Unicode Code Points)
    st.markdown("### Text Length Metrics (`unicode_code_points`)")
    l1, l2, l3, l4, l5 = st.columns(5)
    l1.metric("Min Length", f"{stats.min_length:,}" if stats.min_length is not None else "N/A")
    l2.metric("Max Length", f"{stats.max_length:,}" if stats.max_length is not None else "N/A")
    l3.metric("Mean Length", f"{stats.mean_length:.1f}" if stats.mean_length is not None else "N/A")
    l4.metric("Median Length", f"{stats.median_length:.1f}" if stats.median_length is not None else "N/A")
    l5.metric("IQR (Q3 - Q1)", f"{stats.iqr_length:.1f}" if stats.iqr_length is not None else "N/A")

    st.caption(
        f"**Approximate Whitespace-Delimited Tokens:** Mean = "
        f"`{f'{stats.mean_approx_whitespace_tokens:.1f}' if stats.mean_approx_whitespace_tokens is not None else 'N/A'}`, "
        f"Max = `{stats.max_approx_whitespace_tokens if stats.max_approx_whitespace_tokens is not None else 'N/A'}`. "
        "*Note for Burmese NLP: Burmese script does not place spaces between every grammatical word; "
        "whitespace-delimited counts represent phrase/clause chunks rather than true word counts.*"
    )

    # Section 4: Interactive Plotly Charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Text Length Distribution (Unicode Code Points)")
        if stats.length_histogram:
            bin_labels = [
                f"{b.bin_start:.0f}–{b.bin_end:.0f}" for b in stats.length_histogram
            ]
            bin_counts = [b.count for b in stats.length_histogram]

            fig_hist = go.Figure(
                data=[
                    go.Bar(
                        x=bin_labels,
                        y=bin_counts,
                        marker_color="#75BDF5",
                        marker_line_color="#1F6FB5",
                        marker_line_width=1,
                        hovertemplate="Length Range: %{x} cp<br>Records: %{y:,}<extra></extra>",
                    )
                ]
            )
            fig_hist.update_layout(
                xaxis_title="Length Range (Unicode Code Points)",
                yaxis_title="Record Count",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#23354D", size=12),
                margin=dict(l=16, r=16, t=24, b=40),
                height=340,
            )
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("Insufficient non-empty records to plot length distribution.")

    with chart_col2:
        st.markdown("#### Affected Records by Anomaly Category")
        if stats.affected_records_by_type:
            sorted_types = sorted(
                stats.affected_records_by_type.items(), key=lambda x: x[1]
            )
            y_labels = [k for k, _ in sorted_types]
            x_counts = [v for _, v in sorted_types]
            type_sev_map = {a.issue_type.value: a.severity.value for a in result.anomalies}
            bar_colors = [
                SEVERITY_META[type_sev_map.get(k, Severity.INFO.value)]["bar_color"]
                for k in y_labels
            ]

            fig_anom = go.Figure(
                data=[
                    go.Bar(
                        x=x_counts,
                        y=y_labels,
                        orientation="h",
                        marker_color=bar_colors,
                        hovertemplate="Issue: %{y}<br>Affected Records: %{x:,}<extra></extra>",
                    )
                ]
            )
            fig_anom.update_layout(
                xaxis_title="Unique Affected Records",
                yaxis_title="",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                font=dict(color="#23354D", size=12),
                margin=dict(l=16, r=16, t=24, b=40),
                height=340,
            )
            st.plotly_chart(fig_anom, use_container_width=True, config={"displayModeBar": False})
        else:
            st.caption("No anomalies detected to chart.")


# =====================================================================
# VIEW 5: REPORTS & EXPORTS
# =====================================================================
def render_reports_page(result: AnalysisResult | None) -> None:
    """Render the Reports & Exports download interface."""
    render_page_header(
        "Reports & Exports",
        "Download structured UTF-8 CSV and JSON inspection reports preserving Burmese Unicode and analysis scope metadata.",
    )
    if result is None:
        st.info("Please upload a dataset or select a sample dataset on the **Overview** page first.")
        return

    render_truncation_banner_if_needed(result)
    lr = result.load_result
    stats = result.stats

    st.markdown("### Export Options")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        include_bom = st.checkbox(
            "Include UTF-8 BOM in CSV export (Recommended for Microsoft Excel)",
            value=True,
            help="Adds a 3-byte UTF-8 signature (utf-8-sig) so Excel on Windows opens Burmese script without garbled text.",
        )
    with opt_col2:
        include_full = st.checkbox(
            "Include full original record text in exported findings",
            value=False,
            help=f"By default, reports include bounded excerpts (up to {result.config.excerpt_max_chars} chars) to minimize file size and data exposure.",
        )

    csv_bytes = generate_csv_report(
        result, include_utf8_bom=include_bom, include_full_text=include_full
    )
    json_str = generate_json_report(result, include_full_text=include_full)

    csv_filename = build_export_filename(lr.filename, "csv", result.analysis_timestamp_utc)
    json_filename = build_export_filename(lr.filename, "json", result.analysis_timestamp_utc)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        _render_html(
            f"""
            <div class="kl-panel">
                <strong>CSV Findings Report</strong>
                <p style="font-size:0.88rem;color:#52667A;margin:0.35rem 0;">
                    Contains {stats.total_anomaly_findings:,} consolidated anomaly rows with scope metadata,
                    record indices, line numbers, severity, occurrence counts, and suggested actions.
                </p>
            </div>
            """
        )
        st.download_button(
            label=f"Download CSV Report ({len(csv_bytes) / 1024:.1f} KB)",
            data=csv_bytes,
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True,
        )

    with dl_col2:
        _render_html(
            f"""
            <div class="kl-panel">
                <strong>JSON Full Inspection Report</strong>
                <p style="font-size:0.88rem;color:#52667A;margin:0.35rem 0;">
                    Contains input metadata, truncation notices, configuration, summary statistics,
                    duplicate groups ({stats.duplicate_group_count:,}), and all anomaly findings.
                </p>
            </div>
            """
        )
        st.download_button(
            label=f"Download JSON Report ({len(json_str.encode('utf-8')) / 1024:.1f} KB)",
            data=json_str.encode("utf-8"),
            file_name=json_filename,
            mime="application/json",
            use_container_width=True,
        )

    with st.expander("Preview JSON Report Header & Summary Statistics", expanded=False):
        st.code(json_str[:4000] + ("\n..." if len(json_str) > 4000 else ""), language="json")


# =====================================================================
# VIEW 6: ANALYSIS CONFIGURATION (SETTINGS)
# =====================================================================
def render_settings_page() -> None:
    """Render the Analysis Configuration threshold settings form."""
    render_page_header(
        "Analysis Configuration",
        "Customize statistical outlier thresholds, whitespace rules, and maximum record limits.",
    )
    current_cfg: AnalysisConfig = st.session_state["analysis_config"]

    with st.form("analysis_config_form"):
        st.markdown("### Outlier & Whitespace Thresholds")
        c1, c2 = st.columns(2)
        with c1:
            iqr_mult = st.slider(
                "IQR Outlier Fence Multiplier (k)",
                min_value=0.5,
                max_value=5.0,
                value=float(current_cfg.iqr_multiplier),
                step=0.25,
                help="Fences are computed as Q1 - k*IQR and Q3 + k*IQR. Default 1.5 is standard Tukey fencing; 3.0 flags only extreme outliers.",
            )
            min_iqr_recs = st.number_input(
                "Minimum Non-Empty Records Required for IQR",
                min_value=4,
                max_value=1000,
                value=int(current_cfg.min_records_for_iqr),
                step=1,
                help="Datasets with fewer non-empty records than this threshold skip IQR outlier flagging.",
            )
        with c2:
            rep_space = st.number_input(
                "Repeated Consecutive Space Threshold",
                min_value=2,
                max_value=20,
                value=int(current_cfg.repeated_space_threshold),
                step=1,
                help="Minimum consecutive internal ASCII spaces (U+0020) required to trigger REPEATED_SPACES.",
            )
            excerpt_chars = st.number_input(
                "Maximum Excerpt Length (Characters)",
                min_value=50,
                max_value=2000,
                value=int(current_cfg.excerpt_max_chars),
                step=50,
                help="Maximum character length of text previews stored inside Anomaly objects.",
            )

        st.markdown("### Diagnostic Toggles & Record Limits")
        t1, t2 = st.columns(2)
        with t1:
            detect_norm_dups = st.checkbox(
                "Detect Unicode NFC Normalization Duplicates",
                value=current_cfg.detect_normalized_duplicates,
                help="Flags records that have different raw code points but identical Unicode NFC canonical forms.",
            )
            flag_newlines = st.checkbox(
                "Flag Embedded Newlines (\\r, \\n) Inside Records",
                value=current_cfg.flag_internal_newlines,
                help="Flags multi-line breaks occurring inside a single CSV or JSONL text field.",
            )
            flag_tabs_chk = st.checkbox(
                "Flag Embedded Tab Characters (\\t) Inside Records",
                value=current_cfg.flag_tabs,
                help="Flags tab characters inside records.",
            )
        with t2:
            max_limit = st.number_input(
                "Maximum Analyzed Records Limit (Hard Cap)",
                min_value=10,
                max_value=500_000,
                value=int(current_cfg.max_records_limit),
                step=10_000,
                help="Default 100,000 (configurable up to 500,000). Datasets exceeding this limit are truncated with an explicit scope warning.",
            )

        submitted = st.form_submit_button("Apply Configuration & Re-Analyze", use_container_width=True)
        if submitted:
            st.session_state["analysis_config"] = AnalysisConfig(
                iqr_multiplier=iqr_mult,
                min_records_for_iqr=int(min_iqr_recs),
                repeated_space_threshold=int(rep_space),
                detect_normalized_duplicates=detect_norm_dups,
                flag_internal_newlines=flag_newlines,
                flag_tabs=flag_tabs_chk,
                max_records_limit=int(max_limit),
                excerpt_max_chars=int(excerpt_chars),
            )
            st.success("Analysis configuration updated! Active dataset has been re-analyzed.")
            st.rerun()


# =====================================================================
# MAIN APPLICATION ENTRYPOINT
# =====================================================================
def main() -> None:
    """Configure page layout, sidebar navigation, and active view orchestration."""
    st.set_page_config(
        page_title=f"{__app_name__} — Text Dataset Quality Inspection",
        page_icon=str(BRAND_ICON_PATH) if BRAND_ICON_PATH.exists() else "🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    init_session_state()
    render_sidebar_branding()

    nav_choice = st.sidebar.radio(
        "Navigation",
        options=[
            "Overview",
            "Anomaly Explorer",
            "Unicode Lab",
            "Statistics",
            "Reports & Exports",
            "Analysis Configuration",
        ],
        label_visibility="collapsed",
    )

    # Resolve active dataset & run cached inspection
    file_bytes, filename, _ = _get_active_file_payload()
    result: AnalysisResult | None = None
    schema_info: SchemaInspectionResult | None = None
    load_error: str | None = None

    if file_bytes is not None and filename is not None:
        delim_map = {
            "Auto-detect": None,
            "Comma (,)": ",",
            "Tab (\\t)": "\t",
            "Semicolon (;)": ";",
            "Pipe (|)": "|",
        }
        csv_delim = delim_map.get(st.session_state["csv_delimiter_choice"])
        csv_has_hdr = bool(st.session_state["csv_has_header"])

        schema_info = cached_inspect_schema(
            file_bytes=file_bytes,
            filename=filename,
            explicit_format_str=None,
            csv_delimiter=csv_delim,
            csv_has_header=csv_has_hdr,
        )
        if schema_info.error_message:
            load_error = schema_info.error_message
        else:
            target_field = (
                st.session_state["selected_field_override"]
                or schema_info.suggested_field
            )
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            cfg: AnalysisConfig = st.session_state["analysis_config"]
            try:
                result = cached_load_and_analyze(
                    file_hash=file_hash,
                    file_bytes=file_bytes,
                    filename=filename,
                    config_json=cfg.model_dump_json(),
                    explicit_format_str=None,
                    selected_field=target_field,
                    csv_delimiter=csv_delim,
                    csv_has_header=csv_has_hdr,
                )
            except (UnsupportedFormatError, FileEncodingError) as exc:
                load_error = str(exc)

    # Sidebar Active Dataset Status Pill & Privacy Note
    st.sidebar.markdown("---")
    if result is not None:
        st.sidebar.caption(
            f"**Active File:** `{result.load_result.filename}`  \n"
            f"**Records:** `{result.stats.analyzed_raw_records:,}`  \n"
            f"**Findings:** `{result.stats.total_anomaly_findings:,}`"
        )
    else:
        st.sidebar.caption("**Active File:** None loaded")

    with st.sidebar.expander("Privacy & Data Handling", expanded=False):
        st.caption(
            "**Local Execution:** When run locally (`localhost`), datasets are inspected "
            "entirely in your machine's RAM and never sent to external servers or AI APIs.\n\n"
            "**Hosted Execution:** If deployed to a remote server, uploaded files are processed "
            "in that server's memory for the active session without permanent disk storage."
        )

    # Route to selected page
    if nav_choice == "Overview":
        render_overview_page(result, schema_info, filename, load_error)
    elif nav_choice == "Anomaly Explorer":
        render_anomaly_explorer_page(result)
    elif nav_choice == "Unicode Lab":
        render_unicode_lab_page(result)
    elif nav_choice == "Statistics":
        render_statistics_page(result)
    elif nav_choice == "Reports & Exports":
        render_reports_page(result)
    elif nav_choice == "Analysis Configuration":
        render_settings_page()


if __name__ == "__main__":
    main()