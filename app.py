# app.py — One-Click Full Report (Streamlit)
# ------------------------------------------
# Handles your CSV composed of 6 metric blocks (first table to last table):
# 1) AVERAGE of Course completion %
# 2) AVERAGE of NPS
# 3) SUM of No of Placements(Monthly)
# 4) AVERAGE of Reg to Placement %
# 5) AVERAGE of Active Student %
# 6) AVERAGE of Avg Mentor Rating
#
# Verticals (fixed, column-wise):
# Coding & Development, Commerce, Digital Marketing,
# Hospital Administration, Teaching Skilling, Technical Skilling
#
# Outputs:
# - On-screen dashboard (summary + trends + drilldowns)
# - Single consolidated report (HTML / PDF) with all analyses

import base64, io
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Vertical Health – Full Report", layout="wide")
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
    "AVERAGE of Avg Mentor Rating":     (4.0, 4.5),
}

# -----------------------------
# Helpers
# -----------------------------
def strip_percent_to_float(s):
    if pd.isna(s): return np.nan
    s = str(s).replace("%", "").replace(",", "").strip()
    try: return float(s)
    except: return np.nan

def looks_like_metric_header(cell):
    if pd.isna(cell): return False
    txt = str(cell).strip().lower()
    return any(txt.startswith(m.lower()) for m in METRICS_IN_ORDER)

def best_period_sort_key(series):
    s = series.astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if parsed.notna().any(): return parsed
    mmmYY = pd.to_datetime(s, format="%b%y", errors="coerce")
    if mmmYY.notna().any(): return mmmYY
    nums = pd.to_numeric(s.str.extract(r"(\d+)")[0], errors="coerce")
    return nums

def zone_of(value, red_max, watch_max):
    if pd.isna(value): return "—"
    if value <= red_max: return "🔴 Red"
    if value <= watch_max: return "🟡 Watch"
    return "✅ Healthy"

def fmt_value(metric, v):
    if pd.isna(v): return "—"
    if "%" in metric: return f"{v:.2f}%"
    return f"{v:.2f}"

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
    show_strength_bar = st.checkbox("Include Performance Strength bar chart", value=True)

if up is None:
    st.info("Upload your CSV in the sidebar to generate the full analysis & download the report.")
    st.stop()

# -----------------------------
# Parse CSV into tidy long
# -----------------------------
raw = pd.read_csv(up, header=None)
raw = raw.dropna(how="all").reset_index(drop=True)
if raw.shape[1] > (len(VERTICALS) + 1):
    raw = raw.iloc[:, : (len(VERTICALS) + 1)]

tidy_rows = []
i = 0
while i < len(raw):
    first = raw.iat[i, 0]
    if looks_like_metric_header(first):
        metric = str(first).strip()
        j = i + 1
        while j < len(raw) and not looks_like_metric_header(raw.iat[j, 0]):
            j += 1
        block = raw.iloc[i+1:j].copy()
        if not block.empty:
            block = block.rename(columns={block.columns[0]: "Period"})
            for k, vname in enumerate(VERTICALS, start=1):
                if k < block.shape[1]:
                    block = block.rename(columns={block.columns[k]: vname})
            keep = ["Period"] + [v for v in VERTICALS if v in block.columns]
            block = block[keep]
            long = block.melt(id_vars=["Period"], var_name="Vertical", value_name="Value")
            long["Metric"] = metric
            long["Value"] = long["Value"].apply(strip_percent_to_float)
            long["Period"] = long["Period"].astype(str).str.strip()
            long["Vertical"] = long["Vertical"].astype(str).str.strip()
            tidy_rows.append(long)
        i = j
    else:
        i += 1

if not tidy_rows:
    st.error("No metric blocks detected. Please verify the CSV structure.")
    st.stop()

tidy = pd.concat(tidy_rows, ignore_index=True)

# -----------------------------
# Wide pivot per (Period, Vertical)
# -----------------------------
wide = tidy.pivot_table(index=["Period", "Vertical"], columns="Metric", values="Value", aggfunc="mean").reset_index()
for m in METRICS_IN_ORDER:
    if m not in wide.columns:
        wide[m] = np.nan

# Sort periods
wide["_period_sort"] = best_period_sort_key(wide["Period"])
wide["_row_order"] = np.arange(len(wide))
wide = wide.sort_values(by=["_period_sort", "_row_order"])

# Determine latest and previous period
if wide["_period_sort"].notna().any():
    periods_sorted = wide.sort_values(by=["_period_sort", "_row_order"])["Period"].unique().tolist()
else:
    periods_sorted = wide.sort_values(by="_row_order")["Period"].unique().tolist()

latest_period = periods_sorted[-1]
prev_period = periods_sorted[-2] if len(periods_sorted) >= 2 else None

# -----------------------------
# Summary, Zones, Performance Strength
# -----------------------------
summary_rows = []
strength_rows = []

latest_slice = wide[wide["Period"] == latest_period].copy()

# Normalization for Performance Strength (within latest period, per metric)
norm_maps = {}
for m in METRICS_IN_ORDER:
    s = latest_slice[m].astype(float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        norm_maps[m] = s.apply(lambda _: 0.5)  # uniform score if no variation
    else:
        norm_maps[m] = (s - mn) / (mx - mn)

for v in VERTICALS:
    cur = wide[(wide["Vertical"] == v) & (wide["Period"] == latest_period)]
    if cur.empty: continue
    cur = cur.iloc[0]
    prev = None
    if prev_period is not None:
        prevdf = wide[(wide["Vertical"] == v) & (wide["Period"] == prev_period)]
        prev = prevdf.iloc[0] if not prevdf.empty else None

    rec = {"Vertical": v, "Period": latest_period}
    norm_scores = []

    for m in METRICS_IN_ORDER:
        cur_val = float(cur[m]) if pd.notna(cur[m]) else np.nan
        prev_val = float(prev[m]) if (prev is not None and pd.notna(prev[m])) else np.nan
        delta = cur_val - prev_val if (pd.notna(cur_val) and pd.notna(prev_val)) else np.nan
        z = zone_of(cur_val, *thresholds[m])
        rec[f"{m} (Current)"] = cur_val
        rec[f"{m} (Δ)"] = delta
        rec[f"{m} (Zone)"] = z

        # normalization map lookup (align index by Vertical)
        try:
            idx = latest_slice.index[latest_slice["Vertical"] == v][0]
            ns = norm_maps[m].loc[idx]
        except Exception:
            ns = np.nan
        norm_scores.append(ns)

    valid = [x for x in norm_scores if pd.notna(x)]
    strength_pct = round(np.mean(valid) * 100.0, 1) if valid else np.nan
    rec["Performance Strength %"] = strength_pct
    summary_rows.append(rec)
    strength_rows.append({"Vertical": v, "Performance Strength %": strength_pct})

summary_df = pd.DataFrame(summary_rows)
ranking_df = pd.DataFrame(strength_rows).dropna().sort_values("Performance Strength %", ascending=False)

# Pretty display table
pretty = summary_df.copy()
for m in METRICS_IN_ORDER:
    pretty[f"{m} (Current)"] = pretty[f"{m} (Current)"].apply(lambda v: fmt_value(m, v))
    def _fmt_delta(x, metric=m):
        if pd.isna(x): return "—"
        sign = "+" if x >= 0 else ""
        return f"{sign}{x:.2f}" + ("%" if "%" in metric else "")
    pretty[f"{m} (Δ)"] = pretty[f"{m} (Δ)"].apply(_fmt_delta)

# -----------------------------
# On-screen analysis
# -----------------------------
st.subheader(f"Summary — Latest Period by Vertical  (Latest: {latest_period})")
st.dataframe(pretty, use_container_width=True)
st.caption("Δ = change vs previous period for the same vertical. Zones use the thresholds you set in the sidebar.")

if not ranking_df.empty:
    st.subheader("Performance Strength — Ranking")
    st.dataframe(ranking_df, use_container_width=True)

if not ranking_df.empty and show_strength_bar:
    fig, ax = plt.subplots()
    ax.bar(ranking_df["Vertical"], ranking_df["Performance Strength %"])
    ax.set_ylabel("Performance Strength %")
    ax.set_title(f"Composite Performance Strength (all metrics) — {latest_period}")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=20)
    st.pyplot(fig, use_container_width=True)

st.divider()

st.subheader("Trends by Metric")
metric_choice = st.selectbox("Choose a metric to plot", METRICS_IN_ORDER, index=0)
mdata = tidy[tidy["Metric"] == metric_choice].copy()
if not mdata.empty:
    mdata["_period_sort"] = best_period_sort_key(mdata["Period"])
    mdata["_row_order"] = np.arange(len(mdata))
    mdata = mdata.sort_values(by=["_period_sort", "_row_order"])
    fig, ax = plt.subplots()
    for v in VERTICALS:
        sub = mdata[mdata["Vertical"] == v]
        if sub.empty: continue
        ax.plot(sub["Period"].astype(str).values, sub["Value"].values, marker="o", label=v)
    ax.set_xlabel("Period")
    ax.set_ylabel(metric_choice)
    ax.set_title(f"{metric_choice} — Trend by Vertical")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5)
    st.pyplot(fig, use_container_width=True)
else:
    st.info("No data for that metric.")

st.divider()

st.subheader("Per-Vertical Drilldown")
pick_v = st.multiselect("Choose vertical(s)", VERTICALS)
if pick_v:
    for v in pick_v:
        st.markdown(f"### {v}")
        sub = wide[wide["Vertical"] == v].copy()
        sub = sub.sort_values(by=["_period_sort", "_row_order"])
        st.dataframe(sub[["Period"] + METRICS_IN_ORDER], use_container_width=True)

# -----------------------------
# Report builder (HTML / PDF) — ALL ANALYSIS IN ONE REPORT
# -----------------------------
def _fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig); buf.seek(0)
    return buf.getvalue()

def _metric_trend_figure(metric, tidy_df):
    mdf = tidy_df[tidy_df["Metric"] == metric].copy()
    if mdf.empty: return None
    mdf["_period_sort"] = best_period_sort_key(mdf["Period"])
    mdf["_row_order"] = np.arange(len(mdf))
    mdf = mdf.sort_values(by=["_period_sort", "_row_order"])
    fig, ax = plt.subplots()
    for v in VERTICALS:
        s = mdf[mdf["Vertical"] == v]
        if s.empty: continue
        ax.plot(s["Period"].astype(str).values, s["Value"].values, marker="o", label=v)
    ax.set_title(f"{metric} — Trend by Vertical")
    ax.set_xlabel("Period"); ax.set_ylabel(metric)
    ax.legend(loc="best"); ax.grid(True, linestyle="--", linewidth=0.5)
    return fig

def build_html_report(app_title, latest_period, thresholds, pretty_summary, ranking, tidy_df):
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
    def df_to_html(df, title=None):
        h = []
        if title: h.append(f"<h2>{title}</h2>")
        h.append(df.to_html(index=False, border=0, classes='tbl'))
        return "\n".join(h)

    parts = [css,
        f"<h1>{app_title}</h1>",
        f"<div class='muted'>Generated: {datetime.now():%Y-%m-%d %H:%M} • Latest period: <b>{latest_period}</b></div>",
        "<h2>Executive Summary</h2>",
    ]

    # Executive summary KPIs
    zones_cols = [c for c in pretty_summary.columns if c.endswith("(Zone)")]
    red = (pretty_summary[zones_cols] == "🔴 Red").sum().sum()
    watch = (pretty_summary[zones_cols] == "🟡 Watch").sum().sum()
    healthy = (pretty_summary[zones_cols] == "✅ Healthy").sum().sum()
    parts.append(f"<div class='kpi'>✅ Healthy: <b>{healthy}</b> • 🟡 Watch: <b>{watch}</b> • 🔴 Red: <b>{red}</b></div>")

    parts.append(df_to_html(pretty_summary, "Summary — Latest Period by Vertical"))
    if not ranking.empty:
        parts.append(df_to_html(ranking, "Performance Strength — Ranking"))

    # Trends (all metrics)
    for metric in METRICS_IN_ORDER:
        fig = _metric_trend_figure(metric, tidy_df)
        if fig is None: continue
        png = _fig_to_png_bytes(fig)
        b64 = base64.b64encode(png).decode("utf-8")
        parts.append(f"<div class='section'><h2>{metric} — Trend</h2>"
                     f"<div class='imgwrap'><img src='data:image/png;base64,{b64}' style='max-width:100%;height:auto'/></div></div>")

    return "<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>" + "\n".join(parts) + "</body></html>"

def build_pdf_report(app_title, latest_period, thresholds, pretty_summary, ranking, tidy_df):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = []

    story += [Paragraph(f"<b>{app_title}</b>", styles["Title"]),
              Spacer(1, 6),
              Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M} • Latest period: <b>{latest_period}</b>", styles["Normal"]),
              Spacer(1, 12)]

    # Executive Summary (zone counts)
    zones_cols = [c for c in pretty_summary.columns if c.endswith("(Zone)")]
    red = (pretty_summary[zones_cols] == "🔴 Red").sum().sum()
    watch = (pretty_summary[zones_cols] == "🟡 Watch").sum().sum()
    healthy = (pretty_summary[zones_cols] == "✅ Healthy").sum().sum()
    story += [Paragraph("<b>Executive Summary</b>", styles["Heading2"]),
              Paragraph(f"✅ Healthy: <b>{healthy}</b> • 🟡 Watch: <b>{watch}</b> • 🔴 Red: <b>{red}</b>", styles["Normal"]),
              Spacer(1, 12)]

    # Summary table
    story.append(Paragraph("<b>Summary — Latest Period by Vertical</b>", styles["Heading2"]))
    sdata = [pretty_summary.columns.tolist()] + pretty_summary.values.tolist()
    stab = Table(sdata, repeatRows=1)
    stab.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("FONTSIZE", (0,0), (-1,-1), 7.5),
    ]))
    story += [stab, Spacer(1, 12)]

    # Ranking table
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

    # Metric trends (one section per metric)
    for metric in METRICS_IN_ORDER:
        fig = _metric_trend_figure(metric, tidy_df)
        if fig is None: continue
        png = _fig_to_png_bytes(fig)
        story.append(Paragraph(f"<b>{metric} — Trend</b>", styles["Heading2"]))
        story.append(Image(ImageReader(io.BytesIO(png)), width=720, height=360))
        story.append(PageBreak())

    doc.build(story)
    pdf = buf.getvalue(); buf.close()
    return pdf

st.subheader("📄 Download consolidated report")
app_title = "Vertical Health — Full Analysis Report"
html = build_html_report(app_title, latest_period, thresholds, pretty, ranking_df, tidy)
st.download_button(
    "Download HTML report",
    data=html.encode("utf-8"),
    file_name=f"vertical_full_report_{latest_period}.html",
    mime="text/html"
)

try:
    pdf = build_pdf_report(app_title, latest_period, thresholds, pretty, ranking_df, tidy)
    st.download_button(
        "Download PDF report",
        data=pdf,
        file_name=f"vertical_full_report_{latest_period}.pdf",
        mime="application/pdf"
    )
except Exception as e:
    st.warning(f"PDF generation error: {e}. HTML export still available.")
