# Streamlit Vertical Analysis & Zones — Generic App
# ------------------------------------------------
# Features:
# - Upload CSV (or load a default sample if none is uploaded)
# - Map columns: Vertical, Date (or Week), Metrics (multi-select)
# - Adjustable zone thresholds for each metric
# - Summary KPIs per vertical (latest period, WoW/MoM deltas)
# - Trend charts (one figure per metric, grouped by vertical)
# - Zone tagging: 🔴 Red, 🟡 Watch, ✅ Healthy based on latest period value
# - Export insights as CSV
#
# Notes:
# - Charts use matplotlib only. One chart per figure. No explicit colors set.

import io
import math
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Vertical Analysis & Zones", layout="wide")

st.title("📊 Vertical Analysis & Zone Report")

st.markdown(
    "Upload your CSV and map the columns. Then set thresholds to tag each vertical into **🔴 Red / 🟡 Watch / ✅ Healthy**."
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception:
        file.seek(0)
        return pd.read_excel(file)

def coerce_numeric(series: pd.Series):
    if series.dtype == "object":
        s = series.astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
        return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

def sort_period(col: pd.Series):
    # try to parse as date; if fails, return as-is order
    try:
        parsed = pd.to_datetime(col, errors='raise', dayfirst=False, infer_datetime_format=True)
        return parsed
    except Exception:
        # try to parse like "Week 1", "Week 2", or "YYYY-Wxx"
        # extract trailing numbers for a naive sort
        import re
        idx = col.astype(str).str.extract(r'(\d+)')
        idx = pd.to_numeric(idx[0], errors='coerce')
        return idx

def compute_change(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return np.nan
    return curr - prev

def zone_for_value(val, red_max, watch_max):
    # val <= red_max -> Red; val <= watch_max -> Watch; else Healthy
    if pd.isna(val):
        return "—"
    if val <= red_max:
        return "🔴 Red"
    elif val <= watch_max:
        return "🟡 Watch"
    return "✅ Healthy"

def pct(x):
    if pd.isna(x):
        return "—"
    return f"{x:.2f}%"

def num(x):
    if pd.isna(x):
        return "—"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:.2f}"

# -----------------------------
# Data input
# -----------------------------
with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if up is not None:
        df = load_csv(up)
    else:
        # Create a small sample so the app runs without data
        df = pd.DataFrame({
            "Vertical": ["Coding", "Coding", "Coding", "Commerce", "Commerce", "Technical", "Technical"],
            "Period": ["2025-06-01", "2025-07-01", "2025-08-01", "2025-07-01", "2025-08-01", "2025-07-01", "2025-08-01"],
            "Completion %": [45.0, 52.0, 59.0, 38.0, 42.0, 47.0, 43.0],
            "Engagement %": [62.0, 65.0, 61.0, 58.0, 60.0, 63.0, 64.0],
            "Registrations": [120, 130, 140, 90, 110, 100, 95],
        })

    st.caption(f"Rows loaded: **{len(df)}**")
    if st.checkbox("Preview data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.header("2) Column Mapping")
    cols = list(df.columns)
    col_vertical = st.selectbox("Vertical column", options=cols, index=(cols.index("Vertical") if "Vertical" in cols else 0))
    col_period = st.selectbox("Date/Week/Period column", options=cols, index=(cols.index("Period") if "Period" in cols else 1))
    # Guess numeric metrics
    default_metric_candidates = [c for c in cols if c not in [col_vertical, col_period]]
    metrics = st.multiselect("Metric columns (one or more)", options=cols, default=default_metric_candidates[:3])

    st.header("3) Thresholds & Tags")
    st.caption("Set thresholds for **each metric**. Interpretation: value ≤ Red Max → 🔴 Red; value ≤ Watch Max → 🟡 Watch; else ✅ Healthy.")
    thresholds = {}
    for m in metrics:
        with st.expander(f"Thresholds for: {m}", expanded=False):
            # Guess if metric looks like a percentage
            looks_pct = m.strip().endswith('%') or df[m].astype(str).str.contains('%').any()
            if looks_pct:
                r = st.number_input(f"{m} — Red Max (≤)", value=40.0, key=f"{m}_r")
                w = st.number_input(f"{m} — Watch Max (≤)", value=60.0, key=f"{m}_w")
            else:
                # generic numeric
                r = st.number_input(f"{m} — Red Max (≤)", value=50.0, key=f"{m}_r")
                w = st.number_input(f"{m} — Watch Max (≤)", value=75.0, key=f"{m}_w")
        thresholds[m] = {"red": r, "watch": w}

st.divider()

# -----------------------------
# Cleaning & typing
# -----------------------------
data = df.copy()
# Normalize period sort key
data["_period_sort"] = sort_period(data[col_period])
# Coerce metric columns to numeric
for m in metrics:
    data[m] = coerce_numeric(data[m])

# Drop rows with missing vertical or period
data = data.dropna(subset=[col_vertical, col_period])

# -----------------------------
# Summary table (latest period per vertical)
# -----------------------------
st.subheader("Summary — Latest Period by Vertical")

# Identify latest period overall
# Use the sorted key; if ties/NaN, fallback to group max on the original column
try:
    latest_key = data["_period_sort"].max()
    latest_period_rows = data[data["_period_sort"] == latest_key]
    # If latest_key is NaN (e.g., non-parsable), fallback by last occurrence order
    if len(latest_period_rows) == 0 or pd.isna(latest_key):
        # take last period by appearance per vertical
        latest_period_rows = data.sort_values(by=col_period).groupby(col_vertical, as_index=False).tail(1)
except Exception:
    latest_period_rows = data.sort_values(by=col_period).groupby(col_vertical, as_index=False).tail(1)

# For change calculations, find previous period rows per vertical
data_sorted = data.sort_values(by=["_period_sort", col_period])
prev_rows = data_sorted.groupby(col_vertical).nth(-2).reset_index()

# Build summary
summary_records = []
for _, row in latest_period_rows.iterrows():
    vert = row[col_vertical]
    period_val = row[col_period]
    rec = {"Vertical": vert, "Period": period_val}
    # previous row for same vertical
    prev = prev_rows[prev_rows[col_vertical] == vert]
    for m in metrics:
        curr_val = row[m]
        prev_val = prev[m].iloc[0] if len(prev) else np.nan
        change = compute_change(curr_val, prev_val)
        # Zone
        red_max = thresholds[m]["red"]
        watch_max = thresholds[m]["watch"]
        z = zone_for_value(curr_val, red_max, watch_max)
        rec[f"{m} (Current)"] = curr_val
        rec[f"{m} (Δ)"] = change
        rec[f"{m} (Zone)"] = z
    summary_records.append(rec)

summary_df = pd.DataFrame(summary_records)

# Show summary nicely
st.dataframe(summary_df, use_container_width=True)
st.caption("Δ = change from previous period for the same vertical.")

# -----------------------------
# Trend charts (one per metric)
# -----------------------------
st.subheader("Trends by Metric")

for m in metrics:
    st.markdown(f"**{m}**")
    fig, ax = plt.subplots()
    # Build a pivot: index=period, columns=vertical, values=metric
    pvt = data.pivot_table(index=col_period, columns=col_vertical, values=m, aggfunc='mean')
    # Sort index by our parsed key if possible
    order = sort_period(pvt.index.to_series())
    try:
        pvt = pvt.iloc[order.argsort().values]
    except Exception:
        pass
    # Plot each vertical as a line
    for c in pvt.columns:
        ax.plot(pvt.index.astype(str), pvt[c], marker='o', label=str(c))
    ax.set_xlabel(col_period)
    ax.set_ylabel(m)
    ax.set_title(f"{m} — Trend by Vertical")
    ax.legend(loc="best")
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Per-vertical details
# -----------------------------
st.subheader("Per-Vertical Drilldown")

selected_verticals = st.multiselect(
    "Choose vertical(s) for details", options=sorted(data[col_vertical].dropna().unique().tolist())
)
if selected_verticals:
    for v in selected_verticals:
        st.markdown(f"### {v}")
        dfv = data[data[col_vertical] == v].copy()
        dfv = dfv.sort_values(by=["_period_sort", col_period])
        st.dataframe(dfv[[col_period] + metrics], use_container_width=True)

        # Latest snapshot with zones
        last = dfv.tail(1).squeeze()
        bullet_lines = []
        for m in metrics:
            val = last[m]
            red_max = thresholds[m]["red"]
            watch_max = thresholds[m]["watch"]
            z = zone_for_value(val, red_max, watch_max)
            # format numbers: if metric name has % or all values in range 0-100 -> treat as percent
            is_pct_like = (m.strip().endswith('%') or (dfv[m].dropna().between(0, 100).all()))
            val_txt = f"{val:.2f}{'%' if is_pct_like else ''}" if pd.notna(val) else "—"
            bullet_lines.append(f"- **{m}**: {val_txt} → **{z}**  (Red≤{red_max}, Watch≤{watch_max})")
        st.markdown("\n".join(bullet_lines))

        # Small trend figure for the vertical across metrics (separate charts to honor 1 chart/figure rule)
        for m in metrics:
            fig, ax = plt.subplots()
            ax.plot(dfv[col_period].astype(str).values, dfv[m].values, marker='o')
            ax.set_xlabel(col_period)
            ax.set_ylabel(m)
            ax.set_title(f"{v} — {m} Trend")
            ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
            st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Export Insights
# -----------------------------
st.subheader("Export")
if not summary_df.empty:
    out = summary_df.copy()
    # Attempt to format numeric columns nicely
    for c in out.columns:
        if c.endswith("(Current)") or c.endswith("(Δ)"):
            out[c] = pd.to_numeric(out[c], errors='coerce')
    csv = out.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download insights.csv", data=csv, file_name="vertical_insights.csv", mime="text/csv")
else:
    st.info("No insights to export yet.")
