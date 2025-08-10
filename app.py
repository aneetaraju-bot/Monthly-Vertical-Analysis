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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
    try:
        parsed = pd.to_datetime(col, errors='raise', dayfirst=False, infer_datetime_format=True)
        return parsed
    except Exception:
        idx = col.astype(str).str.extract(r'(\d+)')
        idx = pd.to_numeric(idx[0], errors='coerce')
        return idx

def compute_change(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return np.nan
    return curr - prev

def zone_for_value(val, red_max, watch_max):
    if pd.isna(val):
        return "—"
    if val <= red_max:
        return "🔴 Red"
    elif val <= watch_max:
        return "🟡 Watch"
    return "✅ Healthy"

# -----------------------------
# Data input
# -----------------------------
with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])
    if up is not None:
        df = load_csv(up)
    else:
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
    default_metric_candidates = [c for c in cols if c not in [col_vertical, col_period]]
    metrics = st.multiselect("Metric columns (one or more)", options=cols, default=default_metric_candidates[:3])

    st.header("3) Thresholds & Tags")
    st.caption("Set thresholds for each metric: ≤ Red Max → 🔴 Red; ≤ Watch Max → 🟡 Watch; else ✅ Healthy.")
    thresholds = {}
    for m in metrics:
        with st.expander(f"Thresholds for: {m}", expanded=False):
            looks_pct = m.strip().endswith('%') or df[m].astype(str).str.contains('%').any()
            if looks_pct:
                r = st.number_input(f"{m} — Red Max (≤)", value=40.0, key=f"{m}_r")
                w = st.number_input(f"{m} — Watch Max (≤)", value=60.0, key=f"{m}_w")
            else:
                r = st.number_input(f"{m} — Red Max (≤)", value=50.0, key=f"{m}_r")
                w = st.number_input(f"{m} — Watch Max (≤)", value=75.0, key=f"{m}_w")
        thresholds[m] = {"red": r, "watch": w}

st.divider()

# -----------------------------
# Cleaning & typing
# -----------------------------
data = df.copy()
data["_period_sort"] = sort_period(data[col_period])
for m in metrics:
    data[m] = coerce_numeric(data[m])
data = data.dropna(subset=[col_vertical, col_period])

# -----------------------------
# Summary table (latest period per vertical)
# -----------------------------
st.subheader("Summary — Latest Period by Vertical")
try:
    latest_key = data["_period_sort"].max()
    latest_period_rows = data[data["_period_sort"] == latest_key]
    if len(latest_period_rows) == 0 or pd.isna(latest_key):
        latest_period_rows = data.sort_values(by=col_period).groupby(col_vertical, as_index=False).tail(1)
except Exception:
    latest_period_rows = data.sort_values(by=col_period).groupby(col_vertical, as_index=False).tail(1)

data_sorted = data.sort_values(by=["_period_sort", col_period])
prev_rows = data_sorted.groupby(col_vertical).nth(-2).reset_index()

summary_records = []
for _, row in latest_period_rows.iterrows():
    vert = row[col_vertical]
    period_val = row[col_period]
    rec = {"Vertical": vert, "Period": period_val}
    prev = prev_rows[prev_rows[col_vertical] == vert]
    for m in metrics:
        curr_val = row[m]
        prev_val = prev[m].iloc[0] if len(prev) else np.nan
        change = compute_change(curr_val, prev_val)
        z = zone_for_value(curr_val, thresholds[m]["red"], thresholds[m]["watch"])
        rec[f"{m} (Current)"] = curr_val
        rec[f"{m} (Δ)"] = change
        rec[f"{m} (Zone)"] = z
    summary_records.append(rec)

summary_df = pd.DataFrame(summary_records)
st.dataframe(summary_df, use_container_width=True)
st.caption("Δ = change from previous period for the same vertical.")

# -----------------------------
# Trend charts (one per metric)
# -----------------------------
st.subheader("Trends by Metric")
for m in metrics:
    st.markdown(f"**{m}**")
    fig, ax = plt.subplots()
    pvt = data.pivot_table(index=col_period, columns=col_vertical, values=m, aggfunc='mean')
    order = sort_period(pvt.index.to_series())
    try:
        pvt = pvt.iloc[order.argsort().values]
    except Exception:
        pass
    for c in pvt.columns:
        ax.plot(pvt.index.astype(str), pvt[c], marker='o', label=str(c))
    ax.set_xlabel(col_period)
    ax.set_ylabel(m)
    ax.set_title(f"{m} — Trend by Vertical")
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', linewidth=0.5)
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
        dfv = data[data[col_vertical] == v].sort_values(by=["_period_sort", col_period])
        st.dataframe(dfv[[col_period] + metrics], use_container_width=True)
        last = dfv.tail(1).squeeze()
        bullet_lines = []
        for m in metrics:
            val = last[m]
            z = zone_for_value(val, thresholds[m]["red"], thresholds[m]["watch"])
            is_pct_like = (m.strip().endswith('%') or (dfv[m].dropna().between(0, 100).all()))
            val_txt = f"{val:.2f}{'%' if is_pct_like else ''}" if pd.notna(val) else "—"
            bullet_lines.append(f"- **{m}**: {val_txt} → **{z}**")
        st.markdown("\n".join(bullet_lines))
        for m in metrics:
            fig, ax = plt.subplots()
            ax.plot(dfv[col_period].astype(str), dfv[m], marker='o')
            ax.set_xlabel(col_period)
            ax.set_ylabel(m)
            ax.set_title(f"{v} — {m} Trend")
            ax.grid(True, linestyle='--', linewidth=0.5)
            st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Export Insights
# -----------------------------
st.subheader("Export")
if not summary_df.empty:
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download insights.csv", data=csv, file_name="vertical_insights.csv", mime="text/csv")
else:
    st.info("No insights to export yet.")
