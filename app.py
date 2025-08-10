# app.py — Vertical Health: Full Analysis & One-Click Report
# ----------------------------------------------------------
# Designed for a CSV composed of 6 stacked metric tables (first table to last table):
# 1) AVERAGE of Course completion %
# 2) AVERAGE of NPS
# 3) SUM of No of Placements(Monthly)
# 4) AVERAGE of Reg to Placement %
# 5) AVERAGE of Active Student %
# 6) AVERAGE of Avg Mentor Rating
#
# Verticals across columns (header row of each block):
# Coding & Development, Commerce, Digital Marketing, Hospital Administration, Teaching Skilling, Technical Skilling
#
# Output:
# - Summary: current, Δ vs previous, and Zone per metric for each vertical + Performance Strength %
# - Trends per metric (lines by vertical)
# - Per-vertical drilldowns
# - Download consolidated report (HTML always; PDF if reportlab is available)

import io, re, base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Try to import reportlab (PDF export). If missing, we'll still offer HTML.
REPORTLAB_AVAILABLE = True
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Vertical Health — Full Report", layout="wide")
st.title("📊 Vertical Health — Full Analysis & Report")

# -----------------------------
# Config
# -----------------------------
METRICS_IN_ORDER = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

# Canonical verticals (we will still read names from each block header row)
VERTICALS = [
    "Coding & Development",
    "Commerce",
    "Digital Marketing",
    "Hospital Administration",
    "Teaching Skilling",
    "Technical Skilling",
]

# Default zone thresholds (value ≤ red → Red; value ≤ watch → Watch; else Healthy)
DEFAULT_THRESHOLDS = {
    "AVERAGE of Course completion %":   (50.0, 70.0),
    "AVERAGE of NPS":                   (30.0, 50.0),
    "SUM of No of Placements(Monthly)": (10.0, 20.0),
    "AVERAGE of Reg to Placement %":    (40.0, 60.0),
    "AVERAGE of Active Student %":      (50.0, 70.0),
    "AVERAGE of Avg Mentor Rating":     (4.0, 4.5),  # assumes a 1–5 scale
}

# -----------------------------
# Helpers
# -----------------------------
def to_number(s):
    """Extract the first numeric token from strings like '4.6/5', '4 out of 5', '4,5', '55 %'."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = s.replace(",", ".").replace("%", "")  # normalize commas; strip percent
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except Exception:
        return np.nan

def looks_like_metric_header(cell):
    if pd.isna(cell):
        return False
    txt = str(cell).strip().lower()
    return any(txt.startswith(m.lower()) for m in METRICS_IN_ORDER)

def best_period_sort_key(series):
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if parsed.notna().any():
        return parsed
    mmmYY = pd.to_datetime(s, format="%b%y", errors="coerce")  # Jan25 etc.
    if mmmYY.notna().any():
        return mmmYY
    nums = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    return nums  # may be NaN; we'll fall back to original order

def zone_of(value, red_max, watch_max):
    if pd.isna(value):
        return "—"
    if value <= red_max:
        return "🔴 Red"
    if value <= watch_max:
        return "🟡 Watch"
    return "✅ Healthy"

def fmt_value(metric, v):
    if pd.isna(v):
        return "—"
    if "%" in metric:
        return f"{v:.2f}%"
    return f"{v:.2f}"

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def metric_trend_figure(metric, tidy_df, verticals):
    mdf = tidy_df[tidy_df["Metric"] == metric].copy()
    if mdf.empty:
        return None
    mdf["_period_sort"] = best_period_sort_key(mdf["Period"])
    mdf["_row_order"] = np.arange(len(mdf))
    mdf = mdf.sort_values(by=["_period_sort", "_row_order"])

    fig, ax = plt.subplots()
    for v in verticals:
        s = mdf[mdf["Vertical"] == v]
        if s.empty:
            continue
        ax.plot(s["Period"].astype(str).values, s["Value"].values, marker="o", label=v)
    ax.set_title(f"{metric} — Trend by Vertical")
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5)
    return fig

# -----------------------------
# Sidebar: Upload + thresholds
# -----------------------------
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("Upload the multi-table CSV", type=["csv"])
    st.caption("Expected: 6 blocks from 'AVERAGE of Course completion %' to 'AVERAGE of Avg Mentor Rating'.")

    st.header("2) Thresholds (Zones)")
    thresholds = {}
    for m in METRICS_IN_ORDER:
        r_def, w_def = DEFAULT_THRESHOLDS[m]
        st.markdown(f"**{m}**")
        red = st.number_input(f"Red max (≤) — {m}", value=float(r_def), key=f"{m}_r")
        watch = st.number_input(f"Watch max (≤) — {m}", value=float(w_def), key=f"{m}_w")
        thresholds[m] = (red, watch)

    st.header("3) Report Options")
    include_strength_bar = st.checkbox("Include Performance Strength bar chart", value=True)

if up is None:
    st.info("Upload your CSV in the sidebar to generate the full analysis & download the report.")
    st.stop()

# -----------------------------
# Parse CSV into tidy long (Period | Vertical | Metric | Value)
# -----------------------------
raw = pd.read_csv(up, header=None)
raw = raw.dropna(how="all").reset_index(drop=True)

# Keep at most (1 + #verticals) columns if extra blank columns exist
if raw.shape[1] > (len(VERTICALS) + 1):
    raw = raw.iloc[:, : (len(VERTICALS) + 1)]

tidy_rows = []
i = 0
detected_verticals_union = set()

while i < len(raw):
    header_cell = raw.iat[i, 0]
    if looks_like_metric_header(header_cell):
        metric = str(header_cell).strip()

        # Header row contains vertical names in columns 1..N
        header_row = raw.iloc[i].fillna("").astype(str).str.strip().tolist()
        header_verticals = [v for v in header_row[1:] if v != ""]
        detected_verticals_union.update(header_verticals)

        # Data rows until next metric header (or EOF)
        j = i + 1
        while j < len(raw) and not looks_like_metric_header(raw.iat[j, 0]):
            j += 1

        block = raw.iloc[i+1:j].copy()
        if not block.empty:
            # First column is Period
            block = block.rename(columns={block.columns[0]: "Period"})
            # Rename remaining columns using header_verticals (align by index)
            for k in range(1, min(len(header_verticals) + 1, block.shape[1])):
                block = block.rename(columns={block.columns[k]: header_verticals[k-1]})

            # Keep Period + all header verticals for this block
            keep = ["Period"] + header_verticals[: max(0, block.shape[1] - 1)]
            block = block[keep]

            # Melt to tidy
            long = block.melt(id_vars=["Period"], var_name="Vertical", value_name="Value")
            long["Metric"] = metric
            long["Value"] = long["Value"].apply(to_number)
            long["Period"] = long["Period"].astype(str).str.strip()
            long["Vertical"] = long["Vertical"].astype(str).str.strip()
            long = long.dropna(subset=["Period", "Vertical"])
            tidy_rows.append(long)
        i = j
    else:
        i += 1

if not tidy_rows:
    st.error("No metric blocks detected. Please verify the CSV structure.")
    st.stop()

tidy = pd.concat(tidy_rows, ignore_index=True)

# Use detected verticals (preserve canonical order where possible)
detected_verticals = [v for v in VERTICALS if v in detected_verticals_union] + \
                     [v for v in detected_verticals_union if v not in VERTICALS]

# -----------------------------
# Build wide pivot per (Period, Vertical)
# -----------------------------
wide = tidy.pivot_table(index=["Period", "Vertical"], columns="Metric", values="Value", aggfunc="mean").reset_index()
for m in METRICS_IN_ORDER:
    if m not in wide.columns:
        wide[m] = np.nan

# Sort periods
wide["_period_sort"] = best_period_sort_key(wide["Period"])
wide["_row_order"] = np.arange(len(wide))
wide = wide.sort_values(by=["_period_sort", "_row_order"])

# Determine latest & previous period labels
if wide["_period_sort"].notna().any():
    periods_sorted = wide.sort_values(by=["_period_sort", "_row_order"])["Period"].unique().tolist()
else:
    periods_sorted = wide.sort_values(by="_row_order")["Period
