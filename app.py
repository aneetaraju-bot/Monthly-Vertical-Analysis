# app.py — Vertical Health: Full Analysis & One-Click Report (robust for Sheet13 CSV)
# -----------------------------------------------------------------------------------
# Works with a CSV composed of 6 stacked metric blocks:
# 1) AVERAGE of Course completion %
# 2) AVERAGE of NPS
# 3) SUM of No of Placements(Monthly)
# 4) AVERAGE of Reg to Placement %
# 5) AVERAGE of Active Student %
# 6) AVERAGE of Avg Mentor Rating
#
# Each block's header row:
#   | Metric Name | Coding & Development | Commerce | Digital Marketing | Hospital Administration | Teaching Skilling | Technical Skilling |
# followed by period rows in the first column and values in the rest.

import io
import re
import base64
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---- Optional PDF export (the app still works without it) ----
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
# Configuration
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
    "AVERAGE of Avg Mentor Rating":     (4.0, 4.5),  # 1–5 scale
}

# -----------------------------
# Helpers
# -----------------------------
def to_number(s: object) -> float:
    """Extract first numeric token (handles '4.6/5', '4,6', '55 %', '4 out of 5')."""
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace(",", ".").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else np.nan

def canonicalize_metric_name(raw_text: object) -> str:
    """Map messy header text to the 6 canonical names (tolerates case/space/punctuation)."""
    if raw_text is None:
        return ""
    t = str(raw_text).replace("\xa0", " ").strip()
    t = re.sub(r"\s+", " ", t)
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
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if parsed.notna().any():
        return parsed
    mmmYY = pd.to_datetime(s, format="%b%y", errors="coerce")  # Jan25 style
    if mmmYY.notna().any():
        return mmmYY
    return pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")  # fallback

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
    """Single-metric trend chart; Mentor Rating locked to 0–5 y-axis."""
    mdf = tidy_df[tidy_df["Metric"] == metric].copy()
    if mdf.empty: return None
    mdf["_period_sort"] = best_period_key(mdf["Period"])
    mdf["_row_order"] = np.arange(len(mdf))
    mdf = mdf.sort_values(by=["_period_sort", "_row_order"])

    fig, ax = plt.subplots()
    for v in vertical_order:
        sub = mdf[mdf["Vertical"] == v]
        if sub.empty: continue
        ax.plot(sub["Period"].astype(str).values, sub["Value"].values, marker="o", label=v)
    ax.set_title(f"{metric} — Trend by Vertical")
    ax.set_xlabel("Period")
    ax.set_ylabel(metric)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5)
    if "avg mentor rating" in metric.lower():
        ax.set_ylim(0, 5)
        try: ax.set_yticks(np.arange(0, 5.5, 0.5))
        except Exception: pass
    return fig

# -----------------------------
# Sidebar: upload & thresholds
# -----------------------------
with st.sidebar:
    st.header("1) Upload CSV")
    up = st.file_uploader("Upload the multi-table CSV", type=["csv"])
    st.caption("Sheet13-style CSV with 6 stacked metric blocks; each block's header row lists vertical names.")

    st.header("2) Thresholds (Zones)")
    thresholds = {}
    for m in METRICS:
        r_def, w_def = DEFAULT_THRESHOLDS[m]
        st.markdown(f"**{m}**")
        red = st.number_input(f"Red max (≤) — {m}", value=float(r_def), key=f"{m}_red")
        watch = st.number_input(f"Watch max (≤) — {m}", value=float(w_def), key=f"{m}_watch")
        thresholds[m] = (red, watch)

    st.header("3) Report options")
    show_strength_bar = st.checkbox("Show Performance Strength bar chart", value=True)

if up is None:
    st.info("Upload your CSV to generate the report.")
    st.stop()

# -----------------------------
# Parse the CSV into tidy long
# -----------------------------
raw = pd.read_csv(up, header=None)
raw = raw.dropna(how="all").reset_index(drop=True)

# Trim stray empty columns beyond plausible width
if raw.shape[1] > len(PREFERRED_VERTICALS) + 1:
    raw = raw.iloc[:, : (len(PREFERRED_VERTICALS) + 1)]

tidy_parts = []
i = 0
detected_verticals_union = set()

while i < len(raw):
    head = raw.iat[i, 0]
    if looks_like_metric_header(head):
        metric = canonicalize_metric_name(head)

        header_row = raw.iloc[i].fillna("").astype(str).str.strip().tolist()
        block_verticals = [v for v in header_row[1:] if v != ""]
        detected_verticals_union.update(block_verticals)

        j = i + 1
        while j < len(raw) and not looks_like_metric_header(raw.iat[j, 0]):
            j += 1

        block = raw.iloc[i+1:j].copy()
        if not block.empty:
            block = block.rename(columns={block.columns[0]: "Period"})
            for k in range(1, min(len(block_verticals) + 1, block.shape[1])):
                block = block.rename(columns={block.columns[k]: block_verticals[k-1]})

            keep = ["Period"] + [c for c in block.columns if c != "Period"]
            block = block[keep]

            long = block.melt(id_vars=["Period"], var_name="Vertical", value_name="Value")
            long["Metric"] = metric
            long["Value"] = long["Value"].apply(to_number)
            long["Period"] = long["Period"].astype(str).str.strip()
            long["Vertical"] = long["Vertical"].astype(str).str.strip()
            long = long.dropna(subset=["Period", "Vertical"])
            tidy_parts.append(long)
        i = j
    else:
        i += 1

if not tidy_parts:
    st.error("No metric blocks detected. Please verify the CSV structure.")
    st.stop()

tidy = pd.concat(tidy_parts, ignore_index=True)

# Determine final vertical order
verticals = [v for v in PREFERRED_VERTICALS if v in detected_verticals_union] + \
            [v for v in detected_verticals_union if v not in PREFERRED_VERTICALS]

# -----------------------------
# Wide table per (Period, Vertical)
# -----------------------------
wide = tidy.pivot_table(index=["Period", "Vertical"], columns="Metric",
                        values="Value", aggfunc="mean").reset_index()
for m in METRICS:
    if m not in wide.columns:
        wide[m] = np.nan

# Sort periods robustly
wide["_period_sort"] = best_period_key(wide["Period"])
wide["_row_order"] = np.arange(len(wide))
wide = wide.sort_values(by=["_period_sort", "_row_order"]) if wide["_period_sort"].notna().any() else wide.sort_values(by="_row_order")

periods_sorted = wide["Period"].dropna().astype(str).drop_duplicates().tolist()
if not periods_sorted:
    st.error("No period labels detected. Ensure each block's first column lists periods (e.g., Jan25).")
    st.stop()

latest_period = periods_sorted[-1]
prev_period = periods_sorted[-2] if len(periods_sorted) >= 2 else None

# -----------------------------
# Summary (current, Δ, zone) + Performance Strength
# -----------------------------
summary_rows = []

latest_slice = wide[wide["Period"] == latest_period].copy()
mins = {}; maxs = {}
for m in METRICS:
    s = latest_slice[m].astype(float)
    mins[m], maxs[m] = (s.min(skipna=True), s.max(skipna=True))

for v in verticals:
    cur = wide[(wide["Vertical"] == v) & (wide["Period"] == latest_period)]
    if cur.empty: continue
    cur = cur.iloc[0]
    prev = None
    if prev_period is not None:
        prevdf = wide[(wide["Vertical"] == v) & (wide["Period"] == prev_period)]
        prev = prevdf.iloc[0] if not prevdf.empty else None

    rec = {"Vertical": v, "Period": latest_period}
    norm_scores = []

    for m in METRICS:
        cur_val = float(cur[m]) if pd.notna(cur[m]) else np.nan
        prev_val = float(prev[m]) if (prev is not None and pd.notna(prev[m])) else np.nan
        delta = cur_val - prev_val if (pd.notna(cur_val) and pd.notna(prev_val)) else np.nan
        z = zone_of(cur_val, *thresholds[m])

        rec[f"{m} (Current)"] = cur_val
        rec[f"{m} (Δ)"] = delta
        rec[f"{m} (Zone)"] = z

        mn, mx = mins[m], maxs[m]
        if pd.isna(cur_val) or pd.isna(mn) or pd.isna(mx) or mx == mn:
            ns = 0.5
        else:
            ns = (cur_val - mn) / (mx - mn)
            ns = np.clip(ns, 0, 1)
        norm_scores.append(ns)

    strength_pct = round(np.mean([x for x in norm_scores if pd.notna(x)]) * 100.0, 1) if norm_scores else np.nan
    rec["Performance Strength %"] = strength_pct
    summary_rows.append(rec)

summary_df = pd.DataFrame(summary_rows)

# Guard for empty summary (prevents later KeyErrors)
if summary_df.empty:
    st.error("No data available for the latest period. Check that blocks & periods were parsed.")
    with st.expander("Debug: what got parsed?"):
        try:
            st.write("Metrics:", sorted(tidy["Metric"].dropna().astype(str).str.strip().unique().tolist()))
            st.write("Periods:", sorted(tidy["Period"].dropna().astype(str).str.strip().unique().tolist()))
            st.write("Verticals:", sorted(tidy["Vertical"].dropna().astype(str).str.strip().unique().tolist()))
        except Exception:
            pass
    st.stop()

# Ensure expected columns exist (belt & suspenders)
for base in ["Vertical", "Period"]:
    if base not in summary_df.columns:
        summary_df[base] = np.nan
for m in METRICS:
    for suffix in [" (Current)", " (Δ)", " (Zone)"]:
        col = f"{m}{suffix}"
        if col not in summary_df.columns:
            summary_df[col] = np.nan
if "Performance Strength %" not in summary_df.columns:
    summary_df["Performance Strength %"] = np.nan

# Pretty table for UI/report
pretty = summary_df.copy()
for m in METRICS:
    pretty[f"{m} (Current)"] = pretty[f"{m} (Current)"].apply(lambda v: fmt_value(m, v))
    def _fmt_delta(x, metric=m):
        if pd.isna(x): return "—"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f}" + ("%" if "%" in metric else "")
    pretty[f"{m} (Δ)"] = pretty[f"{m} (Δ)"].apply(_fmt_delta)

# Safe ranking
if {"Vertical", "Performance Strength %"} <= set(summary_df.columns):
    rank = (
        summary_df[["Vertical", "Performance Strength %"]]
        .dropna(subset=["Performance Strength %"])
        .sort_values("Performance Strength %", ascending=False)
    )
else:
    rank = pd.DataFrame(columns=["Vertical", "Performance Strength %"])

# -----------------------------
# On-screen dashboard
# -----------------------------
st.subheader(f"Summary — Latest Period: {latest_period}")
st.dataframe(pretty, use_container_width=True)

if not rank.empty:
    st.subheader("Performance Strength — Ranking")
    st.dataframe(rank, use_container_width=True)
else:
    st.info("No Performance Strength data to rank.")

if st.sidebar.checkbox("Show Performance Strength bar chart", value=True):
    if not rank.empty:
        fig, ax = plt.subplots()
        ax.bar(rank["Vertical"], rank["Performance Strength %"])
        ax.set_ylabel("Performance Strength %")
        ax.set_title(f"Composite Performance Strength — {latest_period}")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        plt.xticks(rotation=20)
        st.pyplot(fig, use_container_width=True)

st.divider()

st.subheader("Trends by Metric")
metric_choice = st.selectbox("Choose metric", METRICS, index=0)
fig = trend_figure(metric_choice, tidy, verticals)
if fig is not None:
    st.pyplot(fig, use_container_width=True)
else:
    st.info("No data for that metric.")

st.divider()

st.subheader("Per-Vertical Drilldown")
pick = st.multiselect("Choose vertical(s)", verticals)
if pick:
    for v in pick:
        st.markdown(f"### {v}")
        sub = wide[wide["Vertical"] == v].copy()
        sub = sub.sort_values(by=["_period_sort", "_row_order"])
        st.dataframe(sub[["Period"] + METRICS], use_container_width=True)

# -----------------------------
# Build consolidated HTML / PDF report
# -----------------------------
def build_html_report(title, latest_period, pretty_summary, ranking, tidy_df):
    css = """
    <style>
      body { font-family: Inter, Arial, sans-serif; margin: 24px; }
      h1 { margin: 0 0 8px 0; }
      .muted { color:#666; margin-bottom: 20px; }
      table.tbl { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
      table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 8px; font-size: 13px; }
      table.tbl th { background:#f7f7f7; text-align:left; }
      .section { margin-top: 24px; }
      .imgwrap { text-align:center; margin: 12px 0 24px; }
      .kpi { margin: 8px 0; }
    </style>
    """
    parts = [
        css,
        f"<h1>{title}</h1>",
        f"<div class='muted'>Generated: {datetime.now():%Y-%m-%d %H:%M} • Latest period: <b>{latest_period}</b></div>",
        "<h2>Executive Summary</h2>",
    ]
    zone_cols = [c for c in pretty_summary.columns if c.endswith("(Zone)")]
    healthy = (pretty_summary[zone_cols] == "✅ Healthy").sum().sum()
    watch = (pretty_summary[zone_cols] == "🟡 Watch").sum().sum()
    red = (pretty_summary[zone_cols] == "🔴 Red").sum().sum()
    parts.append(f"<div class='kpi'>✅ Healthy: <b>{healthy}</b> • 🟡 Watch: <b>{watch}</b> • 🔴 Red: <b>{red}</b></div>")
    parts.append("<h2>Summary — Latest Period by Vertical</h2>")
    parts.append(pretty_summary.to_html(index=False, border=0, classes='tbl'))
    if not ranking.empty:
        parts.append("<h2>Performance Strength — Ranking</h2>")
        parts.append(ranking.to_html(index=False, border=0, classes='tbl'))

    for m in METRICS:
        fig = trend_figure(m, tidy_df, verticals)
        if fig is None: continue
        png = fig_to_png(fig)
        b64 = base64.b64encode(png).decode("utf-8")
        parts.append(f"<div class='section'><h2>{m} — Trend</h2>"
                     f"<div class='imgwrap'><img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto'/></div></div>")
    return "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>" + "\n".join(parts) + "</body></html>"

def build_pdf_report(title, latest_period, pretty_summary, ranking, tidy_df):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"<b>{title}</b>", styles["Title"]),
        Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M} • Latest period: <b>{latest_period}</b>", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("<b>Summary — Latest Period by Vertical</b>", styles["Heading2"]),
    ]
    sdata = [pretty_summary.columns.tolist()] + pretty_summary.values.tolist()
    stab = Table(sdata, repeatRows=1)
    stab.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
    ]))
    story += [stab, Spacer(1, 12)]
    if not ranking.empty:
        story.append(Paragraph("<b>Performance Strength — Ranking</b>", styles["Heading2"]))
        rdata = [ranking.columns.tolist()] + ranking.values.tolist()
        rtab = Table(rdata, repeatRows=1)
        rtab.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story += [rtab, Spacer(1, 12)]
    for m in METRICS:
        fig = trend_figure(m, tidy_df, verticals)
        if fig is None: continue
        png = fig_to_png(fig)
        story.append(Paragraph(f"<b>{m} — Trend</b>", styles["Heading2"]))
        story.append(Image(ImageReader(io.BytesIO(png)), width=720, height=360))
        story.append(PageBreak())
    doc.build(story)
    pdf = buf.getvalue(); buf.close()
    return pdf

# -----------------------------
# Downloads
# -----------------------------
st.subheader("📄 Download consolidated report")
app_title = "Vertical Health — Full Analysis Report"
html = build_html_report(app_title, latest_period, pretty, rank, tidy)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name=f"vertical_full_report_{latest_period}.html",
    mime="text/html"
)

if REPORTLAB:
    try:
        pdf = build_pdf_report(app_title, latest_period, pretty, rank, tidy)
        st.download_button(
            "Download PDF report",
            data=pdf,
            file_name=f"vertical_full_report_{latest_period}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.warning(f"PDF generation error: {e}. HTML export still available.")
else:
    st.info("PDF export requires `reportlab`. Add it to requirements.txt to enable.")

# -----------------------------
# Debug (optional — remove later)
# -----------------------------
with st.expander("🔎 Debug (hide in production)", expanded=False):
    try:
        parsed_metrics = sorted(tidy["Metric"].dropna().astype(str).str.strip().unique().tolist())
        parsed_periods = sorted(tidy["Period"].dropna().astype(str).str.strip().unique().tolist())
        parsed_verticals = sorted(tidy["Vertical"].dropna().astype(str).str.strip().unique().tolist())
    except Exception:
        parsed_metrics, parsed_periods, parsed_verticals = [], [], []
    st.write("Parsed metric names:", parsed_metrics)
    st.write("Parsed periods:", parsed_periods)
    st.write("Detected verticals:", verticals)
    st.write("Latest period:", latest_period, "Prev period:", prev_period)
