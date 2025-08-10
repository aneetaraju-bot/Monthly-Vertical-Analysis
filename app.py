# app.py
# ------------------------------------------------
# Streamlit Vertical Analysis & Zones
# - Upload CSV/Excel (or use bundled/inline demo)
# - Auto-detects "Monthly business Review - Sheet13.csv" wide layout and reshapes to:
#       Period | Vertical | Completion %
# - Map columns: Vertical, Period, Metrics (multi-select)
# - Set Red/Watch/Healthy thresholds per metric
# - Summary (latest period + Δ vs previous + Zone)
# - Trends (one figure per metric; lines by vertical)
# - Per-vertical drilldown (table + mini trend)
# - Export insights as CSV
#
# Requirements (requirements.txt):
# streamlit>=1.33
# pandas>=2.2
# numpy>=1.26
# matplotlib>=3.8
# openpyxl>=3.1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Vertical Analysis & Zone Report", layout="wide")
st.title("📊 Vertical Analysis & Zone Report")

st.markdown(
    "Upload your CSV/Excel and map the columns. Set thresholds to tag each vertical into "
    "**🔴 Red / 🟡 Watch / ✅ Healthy** based on latest period values."
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_csv_or_excel(file) -> pd.DataFrame:
    """Try CSV first, then Excel."""
    try:
        return pd.read_csv(file)
    except Exception:
        try:
            file.seek(0)
        except Exception:
            pass
        return pd.read_excel(file)

def load_demo_df() -> pd.DataFrame:
    """Prefer bundled data/sample.csv if present; else use tiny inline demo."""
    sample_path = Path(__file__).parent / "data" / "sample.csv"
    if sample_path.exists():
        try:
            return pd.read_csv(sample_path)
        except Exception:
            pass
    # inline fallback
    return pd.DataFrame({
        "Vertical": ["Coding","Coding","Coding","Commerce","Commerce","Technical","Technical"],
        "Period":   ["2025-06-01","2025-07-01","2025-08-01","2025-07-01","2025-08-01","2025-07-01","2025-08-01"],
        "Completion %": [45.0, 52.0, 59.0, 38.0, 42.0, 47.0, 43.0],
        "Engagement %": [62.0, 65.0, 61.0, 58.0, 60.0, 63.0, 64.0],
        "Registrations": [120, 130, 140, 90, 110, 100, 95],
    })

def sanitize_columns(df: pd.DataFrame):
    """Flatten MultiIndex headers and deduplicate column names; strip whitespace."""
    # 1) Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for tpl in df.columns:
            parts = [str(x) for x in tpl if pd.notna(x) and str(x).strip() != ""]
            flat.append(" ".join(parts).strip())
        df.columns = flat
    # 2) Strip & force to string
    cols = [str(c).strip() for c in df.columns]
    # 3) Deduplicate
    new_cols, seen, renamed = [], {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_name = f"{c}_{seen[c]}"
            renamed.append((c, new_name))
            new_cols.append(new_name)
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df, renamed

def reshape_mbr_sheet13(df: pd.DataFrame):
    """
    Detect and reshape the 'Monthly business Review - Sheet13.csv' style:
    - Header row label like 'AVERAGE of Course completion % ' in col0.
    - Row 0 contains real headers: 'Helper Date', 'Coding & Development', 'Commerce', etc.
    Returns tidy df with columns: Period, Vertical, Completion %
    """
    first_header = str(df.columns[0]).strip().lower()
    if first_header.startswith("average of course completion"):
        # Row 0 has the real headers
        new_cols = df.iloc[0].astype(str).str.strip().tolist()
        df2 = df.iloc[1:].copy()
        df2.columns = new_cols

        # Normalize period column
        if "Helper Date" in df2.columns:
            df2 = df2.rename(columns={"Helper Date": "Period"})
        else:
            df2 = df2.rename(columns={df2.columns[0]: "Period"})

        id_col = "Period"
        value_cols = [c for c in df2.columns if c != id_col]
        tidy = df2.melt(id_vars=[id_col], value_vars=value_cols,
                        var_name="Vertical", value_name="Completion %")

        # Clean values
        tidy["Completion %"] = (
            tidy["Completion %"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        tidy["Completion %"] = pd.to_numeric(tidy["Completion %"], errors="coerce")
        tidy["Period"] = tidy["Period"].astype(str).str.strip()
        tidy["Vertical"] = tidy["Vertical"].astype(str).str.strip()
        tidy = tidy.dropna(subset=["Period", "Vertical"]).reset_index(drop=True)
        return tidy
    return None  # Not that format

def coerce_numeric(series: pd.Series):
    """Convert a column to numeric, stripping % and commas if needed."""
    if series.dtype == "object":
        s = (
            series.astype(str)
            .str.replace('%', '', regex=False)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        return pd.to_numeric(s, errors='coerce')
    return pd.to_numeric(series, errors='coerce')

def sort_period_key(col: pd.Series):
    """
    Return a sortable key for a period column:
    - Try real dates
    - Else extract numeric token (e.g., 'Week 12' -> 12)
    - Else NaNs (keeps original order fallback)
    """
    parsed = pd.to_datetime(col, errors='coerce')
    if parsed.notna().any():
        return parsed
    idx = col.astype(str).str.extract(r'(\d+)')
    key = pd.to_numeric(idx[0], errors='coerce')
    return key

def compute_change(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return np.nan
    return curr - prev

def zone_for_value(val, red_max, watch_max):
    """
    val ≤ red_max  -> 🔴 Red
    val ≤ watch_max -> 🟡 Watch
    else             -> ✅ Healthy
    """
    if pd.isna(val):
        return "—"
    if val <= red_max:
        return "🔴 Red"
    if val <= watch_max:
        return "🟡 Watch"
    return "✅ Healthy"

# -----------------------------
# Sidebar — data & configuration
# -----------------------------
with st.sidebar:
    st.header("1) Data")
    uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if uploaded is not None:
        raw = load_csv_or_excel(uploaded)
        raw, renamed_cols = sanitize_columns(raw)
        reshaped = reshape_mbr_sheet13(raw)
        df = reshaped if reshaped is not None else raw
        st.success("File uploaded.")
    else:
        raw = load_demo_df()
        raw, renamed_cols = sanitize_columns(raw)
        df = raw
        st.info("No file uploaded — showing demo data. Upload your CSV/Excel to analyze your data.")

    if renamed_cols:
        st.warning(
            "Duplicate/MultiIndex headers normalized: " +
            ", ".join([f"{old} → {new}" for old, new in renamed_cols])
        )

    st.caption(f"Rows loaded: **{len(df)}**")
    if st.checkbox("Preview data"):
        st.dataframe(df.head(25), use_container_width=True)

    st.header("2) Column Mapping")
    cols = list(df.columns)
    if not cols:
        st.error("No columns found in the data.")
        st.stop()

    # Auto-detect if we created a tidy Sheet13
    auto_detected = set(["Period", "Vertical"]).issubset(set(cols))
    completion_exists = "Completion %" in cols

    guess_vertical = "Vertical" if "Vertical" in cols else cols[0]
    guess_period   = "Period" if "Period" in cols else (cols[1] if len(cols) > 1 else cols[0])

    col_vertical = st.selectbox("Vertical column", options=cols, index=cols.index(guess_vertical) if guess_vertical in cols else 0)
    col_period   = st.selectbox("Date/Week/Period column", options=cols, index=cols.index(guess_period) if guess_period in cols else 0)

    metric_candidates = [c for c in cols if c not in [col_vertical, col_period]]
    default_metrics = (["Completion %"] if (auto_detected and completion_exists and "Completion %" in metric_candidates)
                       else (metric_candidates[:3] if metric_candidates else []))
    metrics = st.multiselect("Metric columns (choose one or more)", options=metric_candidates, default=default_metrics)

    st.header("3) Thresholds & Tags")
    st.caption("Set thresholds per metric. Meaning: Value ≤ Red → 🔴; Value ≤ Watch → 🟡; else ✅.")
    thresholds = {}
    for m in metrics:
        with st.expander(f"Thresholds for: {m}", expanded=False):
            looks_pct = m.strip().endswith('%') or df[m].astype(str).str.contains('%').any()
            if looks_pct:
                red = st.number_input(f"{m} — Red Max (≤)", value=40.0, key=f"{m}_r")
                watch = st.number_input(f"{m} — Watch Max (≤)", value=60.0, key=f"{m}_w")
            else:
                red = st.number_input(f"{m} — Red Max (≤)", value=50.0, key=f"{m}_r")
                watch = st.number_input(f"{m} — Watch Max (≤)", value=75.0, key=f"{m}_w")
        thresholds[m] = {"red": red, "watch": watch}

st.divider()

# -----------------------------
# Validate selections
# -----------------------------
if not metrics:
    st.warning("Select at least one metric in the sidebar to begin analysis.")
    st.stop()

# -----------------------------
# Clean & type
# -----------------------------
data = df.copy()

# Preserve input order as secondary sort (useful if period parsing fails)
data["_row_order"] = np.arange(len(data))
data["_period_sort"] = sort_period_key(data[col_period])

# Coerce metrics numeric
for m in metrics:
    data[m] = coerce_numeric(data[m])

# Drop rows missing essential fields
data = data.dropna(subset=[col_vertical, col_period])

if data.empty:
    st.error("No rows left after cleaning. Check your column mapping and missing data.")
    st.stop()

# -----------------------------
# Summary — Latest period by vertical
# -----------------------------
st.subheader("Summary — Latest Period by Vertical")

# Find latest period using parsed key; if fully NaN, fallback to last seen per vertical
if data["_period_sort"].notna().any():
    latest_key = data["_period_sort"].max()
    latest_period_rows = data[data["_period_sort"] == latest_key]
else:
    latest_period_rows = data.sort_values(by="_row_order").groupby(col_vertical, as_index=False).tail(1)

# For Δ, get the previous row per vertical by the sort key (or by order fallback)
data_sorted = (
    data.sort_values(by=["_period_sort", "_row_order"])
    if data["_period_sort"].notna().any()
    else data.sort_values(by="_row_order")
)
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
st.caption("Δ = change from previous period within the same vertical.")

st.divider()

# -----------------------------
# Trends — one chart per metric
# -----------------------------
st.subheader("Trends by Metric")

for m in metrics:
    st.markdown(f"**{m}**")
    # Keep only necessary columns and drop NA in keys
    safe = data[[col_period, col_vertical, m]].dropna(subset=[col_period, col_vertical])
    if safe.empty:
        st.info(f"No valid rows for '{m}' after cleaning.")
        continue

    fig, ax = plt.subplots()

    # pivot: rows = period, columns = vertical, values = metric
    pvt = safe.pivot_table(index=col_period, columns=col_vertical, values=m, aggfunc='mean')

    # sort rows by our parsed key; if all NaN, keep original row order
    order_key = sort_period_key(pvt.index.to_series())
    try:
        pvt = pvt.iloc[order_key.argsort().values] if order_key.notna().any() else pvt
    except Exception:
        pass

    # plot one line per vertical
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
# Per-vertical drilldown
# -----------------------------
st.subheader("Per-Vertical Drilldown")

vertical_options = sorted(data[col_vertical].dropna().astype(str).unique().tolist())
selected_verticals = st.multiselect("Choose vertical(s) for details", options=vertical_options)

if selected_verticals:
    for v in selected_verticals:
        st.markdown(f"### {v}")
        dfv = data[data[col_vertical].astype(str) == v].copy()
        dfv = dfv.sort_values(by=["_period_sort", "_row_order"])
        st.dataframe(dfv[[col_period] + metrics], use_container_width=True)

        # Latest snapshot bullets
        last = dfv.tail(1).squeeze()
        bullets = []
        for m in metrics:
            val = last[m] if m in last else np.nan
            z = zone_for_value(val, thresholds[m]["red"], thresholds[m]["watch"])
            # pct-like formatting if column name ends with % OR values are 0..100
            is_pct_like = (m.strip().endswith('%') or (dfv[m].dropna().between(0, 100).all()))
            val_txt = f"{val:.2f}{'%' if is_pct_like else ''}" if pd.notna(val) else "—"
            bullets.append(f"- **{m}**: {val_txt} → **{z}**  (Red≤{thresholds[m]['red']}, Watch≤{thresholds[m]['watch']})")
        st.markdown("\n".join(bullets))

        # Mini trends (one chart per metric)
        for m in metrics:
            fig, ax = plt.subplots()
            ax.plot(dfv[col_period].astype(str).values, dfv[m].values, marker='o')
            ax.set_xlabel(col_period)
            ax.set_ylabel(m)
            ax.set_title(f"{v} — {m} Trend")
            ax.grid(True, linestyle='--', linewidth=0.5)
            st.pyplot(fig, use_container_width=True)

st.divider()

# -----------------------------
# Export insights
# -----------------------------
st.subheader("Export")
if not summary_df.empty:
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download insights.csv", data=csv, file_name="vertical_insights.csv", mime="text/csv")
else:
    st.info("No insights to export yet. Upload data and map at least one metric.")
