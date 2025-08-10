# app.py — Multi‑Metric Vertical Health Report (Streamlit)
# --------------------------------------------------------
# Handles a CSV made of multiple blocks, each block shaped like:
#   [Metric Name, V1, V2, ... V6]
#   [Period,     val, val, ...  ]
#   ...
# For your metrics (in order, first table to last table):
#   1) AVERAGE of Course completion %
#   2) AVERAGE of NPS
#   3) SUM of No of Placements(Monthly)
#   4) AVERAGE of Reg to Placement %
#   5) AVERAGE of Active Student %
#   6) AVERAGE of Avg Mentor Rating
#
# Verticals (columns across):
#   Coding & Development, Commerce, Digital Marketing,
#   Hospital Administration, Teaching Skilling, Technical Skilling
#
# Features:
# - Auto-detect & reshape block tables to tidy long form
# - Trend (current vs previous) + Zone (Red/Watch/Healthy)
# - Performance Strength (composite across all metrics)
# - Summary table + trend charts + export CSV
#
# Notes:
# - Charts: matplotlib only, one chart per figure, no custom colors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import StringIO

st.set_page_config(page_title="Vertical Health Report", layout="wide")
st.title("📊 Vertical Health Report — Multi‑Metric (Vertical‑wise)")

# -----------------------------
# Configuration (metric names & verticals)
# -----------------------------
METRICS_IN_ORDER = [
    "AVERAGE of Course completion %",
    "AVERAGE of NPS",
    "SUM of No of Placements(Monthly)",
    "AVERAGE of Reg to Placement %",
    "AVERAGE of Active Student %",
    "AVERAGE of Avg Mentor Rating",
]

VERTICALS = [
    "Coding & Development",
    "Commerce",
    "Digital Marketing",
    "Hospital Administration",
    "Teaching Skilling",
    "Technical Skilling",
]

# Default thresholds (value ≤ red → 🔴, value ≤ watch → 🟡, else ✅)
DEFAULT_THRESHOLDS = {
    "AVERAGE of Course completion %":   (50.0, 70.0),
    "AVERAGE of NPS":                   (30.0, 50.0),
    "SUM of No of Placements(Monthly)": (10.0, 20.0),
    "AVERAGE of Reg to Placement %":    (40.0, 60.0),
    "AVERAGE of Active Student %":      (50.0, 70.0),
    "AVERAGE of Avg Mentor Rating":     (4.0, 4.5),
}

ZONE_POINTS = {"🔴 Red": 0, "🟡 Watch": 1, "✅ Healthy": 2}

# -----------------------------
# Helpers
# -----------------------------
def strip_percent_to_float(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    s = s.replace("%", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan

def normalize_metric_name(x):
    return str(x).strip()

def looks_like_metric_header(cell):
    if pd.isna(cell):
        return False
    txt = str(cell).strip()
    return any(txt.lower().startswith(m.lower()) for m in METRICS_IN_ORDER)

def best_period_sort_key(series):
    # Handles formats like 2025-08-01, "Jan25", "Week 3", etc.
    s = series.astype(str).str.strip()

    # Try real dates first
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if parsed.notna().any():
        return parsed

    # Try MMMYY (e.g., Jan25)
    try_mmmYY = pd.to_datetime(s, format="%b%y", errors="coerce")
    if try_mmmYY.notna().any():
        return try_mmmYY

    # Extract numeric token (e.g., "Week 12" -> 12)
    nums = s.str.extract(r"(\d+)")[0]
    nums = pd.to_numeric(nums, errors="coerce")
    return nums  # may be all NaN; we’ll use original order as fallback

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
    if "Rating" in metric or "NPS" in metric or "Placements" in metric:
        return f"{v:.2f}"
    # Treat percent-looking metrics with % suffix
    if "%" in metric:
        return f"{v:.2f}%"
    return f"{v:.2f}"

# -----------------------------
# Sidebar: Upload & thresholds
# -----------------------------
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("Upload the multi-block CSV", type=["csv"])
    st.caption("Expected: six blocks from ‘AVERAGE of Course completion %’ to ‘AVERAGE of Avg Mentor Rating’.")
    st.header("2) Thresholds")
    thresholds = {}
    for m in METRICS_IN_ORDER:
        red_def, watch_def = DEFAULT_THRESHOLDS[m]
        st.markdown(f"**{m}**")
        red = st.number_input(f"Red max (≤) — {m}", value=float(red_def), key=f"{m}_red")
        watch = st.number_input(f"Watch max (≤) — {m}", value=float(watch_def), key=f"{m}_watch")
        thresholds[m] = (red, watch)
    st.header("3) Options")
    show_strength_bar = st.checkbox("Show Performance Strength bar chart", value=True)

# -----------------------------
# Load & reshape CSV blocks
# -----------------------------
if up is None:
    st.info("Upload your CSV in the sidebar to generate the detailed report.")
    st.stop()

raw_df = pd.read_csv(up, header=None)  # read as raw cells (no header)
# Remove fully empty rows/cols
raw_df = raw_df.dropna(how="all").reset_index(drop=True)
# Try to keep only the first len(VERTICALS)+1 columns if extra empty "Unnamed" columns exist
if raw_df.shape[1] > (len(VERTICALS) + 1):
    raw_df = raw_df.iloc[:, : (len(VERTICALS) + 1)]

# Parse blocks
tidy_rows = []  # rows: Period | Vertical | Metric | Value(float)
i = 0
while i < len(raw_df):
    first_cell = raw_df.iat[i, 0]
    if looks_like_metric_header(first_cell):
        # Metric header row; columns 1..N should be vertical names
        metric_name = normalize_metric_name(first_cell)
        # Use the known vertical names to align columns; if the row has them, fine; else fallback to VERTICALS list
        header_row = raw_df.iloc[i]
        # Expected: header_row[1:] matches VERTICALS (or close)
        # Move to data rows until next metric header or end
        j = i + 1
        while j < len(raw_df) and not looks_like_metric_header(raw_df.iat[j, 0]):
            j += 1
        data_block = raw_df.iloc[i+1:j].copy()
        # First col in data_block is Period
        data_block = data_block.rename(columns={data_block.columns[0]: "Period"})
        # Map other columns to verticals; if missing names, assume the canonical VERTICALS order
        # If header row actually contains vertical names in cols 1..:
        possible_names = [str(x).strip() if pd.notna(x) else "" for x in header_row.tolist()[1:1+len(VERTICALS)]]
        if all(name for name in possible_names):
            col_map = {}
            for idx, name in enumerate(possible_names, start=1):
                col_map[data_block.columns[idx]] = name
            data_block = data_block.rename(columns=col_map)
        else:
            # Force rename to known VERTICALS
            for k, vname in enumerate(VERTICALS, start=1):
                if k < data_block.shape[1]:
                    data_block = data_block.rename(columns={data_block.columns[k]: vname})

        # Keep only Period + our known verticals (if present)
        keep_cols = ["Period"] + [v for v in VERTICALS if v in data_block.columns]
        data_block = data_block[keep_cols]

        # Melt to long
        long = data_block.melt(id_vars=["Period"], var_name="Vertical", value_name="Value")
        long["Metric"] = metric_name
        # Clean value
        long["Value"] = long["Value"].apply(strip_percent_to_float)
        long["Period"] = long["Period"].astype(str).str.strip()
        long["Vertical"] = long["Vertical"].astype(str).str.strip()
        # Drop empty
        long = long.dropna(subset=["Period", "Vertical"])
        tidy_rows.append(long)

        i = j  # continue after this block
    else:
        i += 1

if not tidy_rows:
    st.error("Could not detect any metric blocks. Please confirm the CSV format matches the expected multi-table structure.")
    st.stop()

tidy = pd.concat(tidy_rows, ignore_index=True)

# -----------------------------
# Build wide per-metric columns (Period, Vertical + each Metric as column)
# -----------------------------
pivot = tidy.pivot_table(index=["Period", "Vertical"], columns="Metric", values="Value", aggfunc="mean").reset_index()
# Ensure all metrics exist as columns (even if missing in data)
for m in METRICS_IN_ORDER:
    if m not in pivot.columns:
        pivot[m] = np.nan

# Period sorting key
pivot["_period_sort"] = best_period_sort_key(pivot["Period"])
# If completely NaN, preserve original order by an index key
pivot["_row_order"] = np.arange(len(pivot))
pivot = pivot.sort_values(by=["_period_sort", "_row_order"])

# -----------------------------
# Compute current vs previous per vertical & zones
# -----------------------------
summary_records = []
strength_records = []

for vert in VERTICALS:
    sub = pivot[pivot["Vertical"] == vert].copy()
    if sub.empty:
        continue
    # determine latest & previous
    if sub["_period_sort"].notna().any():
        sub = sub.sort_values(by=["_period_sort", "_row_order"])
    else:
        sub = sub.sort_values(by="_row_order")

    latest = sub.tail(1).squeeze()
    prev = sub.tail(2).head(1).squeeze() if len(sub) >= 2 else None

    row = {"Vertical": vert, "Period": latest["Period"] if "Period" in latest else "—"}
    zones_for_strength = []
    for m in METRICS_IN_ORDER:
        cur = latest.get(m, np.nan)
        prv = prev.get(m, np.nan) if prev is not None else np.nan
        delta = cur - prv if (pd.notna(cur) and pd.notna(prv)) else np.nan
        red_max, watch_max = thresholds[m]
        z = zone_of(cur, red_max, watch_max)
        row[f"{m} (Current)"] = cur
        row[f"{m} (Δ)"] = delta
        row[f"{m} (Zone)"] = z
        zones_for_strength.append(z)
    # Performance Strength
    pts = sum(ZONE_POINTS.get(z, 0) for z in zones_for_strength)
    strength_pct = round((pts / (len(METRICS_IN_ORDER) * 2)) * 100, 1)
    row["Performance Strength %"] = strength_pct
    summary_records.append(row)
    strength_records.append({"Vertical": vert, "Performance Strength %": strength_pct})

summary_df = pd.DataFrame(summary_records)
strength_df = pd.DataFrame(strength_records).sort_values("Performance Strength %", ascending=False)

# -----------------------------
# Show Summary
# -----------------------------
st.subheader("Summary — Latest Period by Vertical (All Metrics)")
if summary_df.empty:
    st.warning("No summary available (insufficient data).")
else:
    # Pretty formatting for display
    display_df = summary_df.copy()
    for m in METRICS_IN_ORDER:
        # Format values
        display_df[f"{m} (Current)"] = display_df[f"{m} (Current)"].apply(lambda v: fmt_value(m, v))
        display_df[f"{m} (Δ)"] = display_df[f"{m} (Δ)"].apply(lambda v: "—" if pd.isna(v) else f"{v:+.2f}" + ("%" if "%" in m else ""))
    st.dataframe(display_df, use_container_width=True)
    st.caption("Δ = change vs previous period for the same vertical. Zones use your thresholds.")

# -----------------------------
# Performance Strength Chart (optional)
# -----------------------------
if show_strength_bar and not strength_df.empty:
    st.subheader("Performance Strength by Vertical")
    fig, ax = plt.subplots()
    ax.bar(strength_df["Vertical"].astype(str), strength_df["Performance Strength %"])
    ax.set_ylabel("Performance Strength %")
    ax.set_title("Composite Performance Strength (all metrics)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=20)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Trends per Metric
# -----------------------------
st.subheader("Trends by Metric")
metric_choice = st.selectbox("Choose a metric to plot trends", METRICS_IN_ORDER, index=0)

# Build a tidy slice for the chosen metric
mdata = tidy[tidy["Metric"] == metric_choice].copy()
if mdata.empty:
    st.info("No data for the selected metric.")
else:
    # Sort by period key
    mdata["_period_sort"] = best_period_sort_key(mdata["Period"])
    mdata["_row_order"] = np.arange(len(mdata))
    mdata = mdata.sort_values(by=["_period_sort", "_row_order"])

    fig, ax = plt.subplots()
    # For each vertical, plot line of Value over Period
    for v in VERTICALS:
        sub = mdata[mdata["Vertical"] == v]
        if sub.empty:
            continue
        ax.plot(sub["Period"].astype(str).values, sub["Value"].values, marker="o", label=v)
    ax.set_xlabel("Period")
    ax.set_ylabel(metric_choice)
    ax.set_title(f"{metric_choice} — Trend by Vertical")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Drilldown (per vertical all metrics)
# -----------------------------
st.subheader("Per-Vertical Drilldown")
pick_verticals = st.multiselect("Choose vertical(s)", VERTICALS)
if pick_verticals:
    for v in pick_verticals:
        st.markdown(f"### {v}")
        sub = pivot[pivot["Vertical"] == v].copy()
        # Sort
        if sub["_period_sort"].notna().any():
            sub = sub.sort_values(by=["_period_sort", "_row_order"])
        else:
            sub = sub.sort_values(by="_row_order")
        # Show table
        show_cols = ["Period"] + METRICS_IN_ORDER
        st.dataframe(sub[show_cols], use_container_width=True)

        # Bullet: current zones
        latest = sub.tail(1).squeeze()
        bullets = []
        for m in METRICS_IN_ORDER:
            val = latest.get(m, np.nan)
            red_max, watch_max = thresholds[m]
            z = zone_of(val, red_max, watch_max)
            bullets.append(f"- **{m}**: {fmt_value(m, val)} → **{z}**  (Red≤{red_max}, Watch≤{watch_max})")
        st.markdown("\n".join(bullets))

st.divider()

# -----------------------------
# Export
# -----------------------------
st.subheader("Export")
if not summary_df.empty:
    csv = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download summary_insights.csv", data=csv, file_name="summary_insights.csv", mime="text/csv")
else:
    st.info("No insights to export yet.")
