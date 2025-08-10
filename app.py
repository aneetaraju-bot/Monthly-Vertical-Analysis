# app.py — Vertical Health: Full Analysis & Report
# Robust for Sheet13 CSV: metric header may be in ANY column; verticals on same row OR next row.

import io, re, base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Optional PDF support
REPORTLAB = True
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB = False

st.set_page_config(page_title="Vertical Health — Full Report", layout="wide")
st.title("📊 Vertical Health — Full Analysis & Report")

# -----------------------------
# Config
# -----------------------------
METRICS = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

PREFERRED_VERTICALS = [
    "Coding & Development",
    "Commerce",
    "Digital Marketing",
    "Hospital Administration",
    "Teaching Skilling",
    "Technical Skilling",
]

DEFAULT_THRESHOLDS = {
    "AVERAGE of Course completion %":   (50.0, 70.0),
    "AVERAGE of NPS":                   (30.0, 50.0),
    "SUM of No of Placements(Monthly)": (10.0, 20.0),
    "AVERAGE of Reg to Placement %":    (40.0, 60.0),
    "AVERAGE of Active Student %":      (50.0, 70.0),
    "AVERAGE of Avg Mentor Rating":     (4.0, 4.5),  # assumes 1–5
}

# -----------------------------
# Helpers
# -----------------------------
def to_number(s: object) -> float:
    """Extract first numeric token: handles '4.6/5', '4,6', '55 %', '4 out of 5'."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace(",", ".").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan

def canonicalize_metric_name(raw_text: object) -> str:
    """Map messy header text to the 6 canonical metric names."""
    if raw_text is None:
        return ""
    t = str(raw_text).replace("\xa0"," ").strip()
    t = re.sub(r"\s+"," ", t)
    low = t.lower().replace("‐","-").replace("–","-").replace("—","-")
    key_map = {
        "average of course completion %": "AVERAGE of Course completion %",
        "average of nps": "AVERAGE of NPS",
        "sum of no of placements(monthly)": "SUM of No of Placements(Monthly)",
        "average of reg to placement %": "AVERAGE of Reg to Placement %",
        "average of active student %": "AVERAGE of Active Student %",
        "average of avg mentor rating": "AVERAGE of Avg Mentor Rating",
    }
    if low in key_map:
        return key_map[low]
    try:
        from difflib import get_close_matches
        match = get_close_matches(low, list(key_map.keys()), n=1, cutoff=0.85)
        if match:
            return key_map[match[0]]
    except Exception:
        pass
    return t

def looks_like_metric_header(cell: object) -> bool:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return False
    return canonicalize_metric_name(cell) in METRICS

def best_period_key(series: pd.Series):
    """Return sortable key for mixed period labels."""
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if parsed.notna().any():
        return parsed
    mmmYY = pd.to_datetime(s, format="%b%y", errors="coerce")
    if mmmYY.notna().any():
        return mmmYY
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")

def zone_of(value: float, red_max: float, watch_max: float) -> str:
    if pd.isna(value): return "—"
    if value <= red_max: return "🔴 Red"
    if value <= watch_max: return "🟡 Watch"
    return "✅ Healthy"

def fmt_value(metric: str, v: float) -> str:
    if pd.isna(v): return "—"
    return f"{v:.2f}%" if "%" in metric else f"{v:.2f}"

def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig); buf.seek(0)
    return buf.getvalue()

def trend_figure(metric: str, tidy_df: pd.DataFrame, vertical_order: list):
    """Single-metric trend; Mentor Rating locked to 0–5 y-axis."""
    mdf = tidy_df[tidy_df["Metric"] == metric].copy()
    if mdf.empty:
        return None
    mdf["_period_sort"] = best_period_key(mdf["Period"])
    mdf["_row_order"] = np.arange(len(mdf))
    mdf = mdf.sort_values(by=["_period_sort", "_row_order"])

    fig, ax = plt.subplots()
    for v in vertical_order:
        sub = mdf[mdf["Vertical"] == v]
        if sub.empty:
            continue
        ax.plot(sub["Period"].astype(str).values, sub["Value"].values, marker="o", label=v)
    ax.set_title(f"{metric} — Trend by Vertical")
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5)

    if "avg mentor rating" in metric.lower():
        ax.set_ylim(0, 5)
        try:
            ax.set_yticks(np.arange(0, 5.5, 0.5))
        except Exception:
            pass
    return fig

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("Upload the multi-table CSV", type=["csv"])
    st.caption("Metric header may be in ANY column; vertical names can be on the same row OR the next row.")

    st.header("2) Thresholds (Zones)")
    thresholds = {}
    for m in METRICS:
        r_def, w_def = DEFAULT_THRESHOLDS[m]
        st.markdown(f"**{m}**")
        red = st.number_input(f"Red max (≤) — {m}", value=float(r_def), key=f"{m}_r")
        watch = st.number_input(f"Watch max (≤) — {m}", value=float(w_def), key=f"{m}_w")
        thresholds[m] = (red, watch)

    st.header("3) Report options")
    show_strength_bar = st.checkbox("Show Performance Strength bar chart", value=True)

if up is None:
    st.info("Upload your CSV to generate the report.")
    st.stop()

# -----------------------------
# Parse CSV (position/layout agnostic)
# -----------------------------
raw = pd.read_csv(up, header=None)
raw = raw.dropna(how="all").reset_index(drop=True)
n_rows, n_cols = raw.shape

def row_metric_info(row_vals):
    """If any cell in a row matches a metric header, return (col_index, canonical_metric)."""
    for ci, cell in enumerate(row_vals):
        if pd.isna(cell):
            continue
        name = canonicalize_metric_name(cell)
        if name in METRICS:
            return ci, name
    return None, None

tidy_parts = []
detected_verticals_union = set()
r = 0

while r < n_rows:
    row_vals = raw.iloc[r].fillna("").astype(str).str.strip().tolist()
    start_col, metric = row_metric_info(row_vals)
    if metric is None:
        r += 1
        continue

    # SAME-ROW header (right of metric cell)
    same_row_cells = raw.iloc[r, start_col+1:n_cols].fillna("").astype(str).str.strip().tolist()
    same_row_verticals = [v for v in same_row_cells if v != ""]

    # NEXT-ROW header (right of metric cell)
    next_row_verticals = []
    if r + 1 < n_rows:
        next_row_cells = raw.iloc[r+1, start_col+1:n_cols].fillna("").astype(str).str.strip().tolist()
        next_row_verticals = [v for v in next_row_cells if v != ""]

    header_verticals = same_row_verticals if same_row_verticals else next_row_verticals
    header_is_next_row = (not same_row_verticals and bool(next_row_verticals))
    if not header_verticals:
        r += 1
        continue

    detected_verticals_union.update(header_verticals)

    # Find next metric header row
    r2 = r + 1
    while r2 < n_rows:
        nxt_vals = raw.iloc[r2].fillna("").astype(str).str.strip().tolist()
        cidx, mname = row_metric_info(nxt_vals)
        if mname is not None:
            break
        r2 += 1

    # Data starts one row below header; if header verticals are on next row, data starts at r+2
    data_start = r + 2 if header_is_next_row else r + 1
    if data_start >= r2:
        r = r2
        continue

    # Slice columns: [start_col .. start_col + 1 + len(header_verticals)]
    right_bound = min(n_cols, start_col + 1 + len(header_verticals))
    block = raw.iloc[data_start:r2, start_col:right_bound].copy()

    # Rename columns
    col_map = {block.columns[0]: "Period"}
    for j, vname in enumerate(header_verticals, start=1):
        if j < block.shape[1]:
            col_map[block.columns[j]] = vname
    block = block.rename(columns=col_map)

    # Keep Period + present verticals
    keep_cols = ["Period"] + [c for c in block.columns if c != "Period"]
    block = block[keep_cols]

    # Melt to tidy long
    long = block.melt(id_vars=["Period"], var_name="Vertical", value_name="Value")
    long["Metric"] = metric
    long["Value"] = long["Value"].apply(to_number)
    long["Period"] = long["Period"].astype(str).str.strip()
    long["Vertical"] = long["Vertical"].astype(str).str.strip()
    long = long.dropna(subset=["Period", "Vertical"])
    tidy_parts.append(long)

    # Jump to next header
    r = r2

if not tidy_parts:
    st.error("No metric blocks detected. Ensure each metric header appears in a row and its vertical names are to the RIGHT (same row or next row).")
    with st.expander("Debug: CSV peek"):
        st.dataframe(raw.head(40))
    st.stop()

tidy = pd.concat(tidy_parts, ignore_index=True)

# Determine vertical order (preferred first, then extras)
verticals = [v for v in PREFERRED_VERTICALS if v in detected_verticals_union] + \
            [v for v in detected_verticals_union if v not in PREFERRED_VERTICALS]

# -----------------------------
# Wide pivot & period ordering
# -----------------------------
wide = tidy.pivot_table(index=["Period","Vertical"], columns="Metric", values="Value", aggfunc="mean").reset_index()
for m in METRICS:
    if m not in wide.columns:
        wide[m] = np.nan

wide["_period_sort"] = best_period_key(wide["Period"])
wide["_row_order"] = np.arange(len(wide))
if wide["_period_sort"].notna().any():
    wide = wide.sort_values(by=["_period_sort","_row_order"])
else:
    wide = wide.sort_values(by="_row_order")

periods_sorted = wide["Period"].dropna().astype(str).drop_duplicates().tolist()
if not periods_sorted:
    st.error("No period labels detected. The 'Period' column under each metric must contain values (e.g., Jan25).")
    st.stop()

latest_period = periods_sorted[-1]
prev_period = periods_sorted[-2] if len(periods_sorted) >= 2 else None

# -----------------------------
# Summary + Performance Strength
# -----------------------------
summary_rows = []
latest_slice = wide[wide["Period"] == latest_period].copy()

mins, maxs = {}, {}
for m in METRICS:
    s = latest_slice[m].astype(float)
    mins[m], maxs[m] = (s.min(skipna=True), s.max(skipna=True))

for v in verticals:
    cur = wide[(wide["Vertical"] == v) & (wide["Period"] == latest_period)]
    if cur.empty:
        continue
    cur = cur.iloc[0]
    prev = None
    if prev_period is not None:
        prevdf = wide[(wide["Vertical"] == v) & (wide["Period"] == prev_period)]
        prev = prevdf.iloc[0] if not prevdf.empty else None

    rec, norm_scores = {"Vertical": v, "Period": latest_period}, []

    for m in METRICS:
        cur_val = float(cur[m]) if pd.notna(cur[m]) else np.nan
        prev_val = float(prev[m]) if (prev is not None and pd.notna(prev[m])) else np.nan
        delta = cur_val - prev_val if (pd.notna(cur_val) and pd.notna(prev_val)) else np.nan
        z = zone_of(cur_val, *thresholds[m])

        rec[f"{m} (Current)"] = cur_val
        rec[f"{m} (Δ)"] = delta
        rec[f"{m} (Zone)"] =
