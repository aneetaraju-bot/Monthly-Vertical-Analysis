import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --------------------------
# SETTINGS
# --------------------------
VERTICALS = [
    "Coding & Development",
    "Commerce",
    "Digital Marketing",
    "Hospital Administration",
    "Teaching Skilling",
    "Technical Skilling"
]

METRIC_WEIGHTS = {
    "AVERAGE of Course completion %": 1,
    "AVERAGE of NPS": 1,
    "SUM of No of Placements(Monthly)": 1,
    "AVERAGE of Reg to Placement %": 1,
    "AVERAGE of Active Student %": 1,
    "AVERAGE of Avg Mentor Rating": 1
}

# --------------------------
# HELPERS
# --------------------------
def looks_like_metric_header(cell):
    if not isinstance(cell, str):
        return False
    keywords = ["course completion", "nps", "placements", "mentor rating", "active student", "reg to placement"]
    return any(k.lower() in cell.lower() for k in keywords)

def to_number(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    s = s.replace(",", ".").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not m:
        return np.nan
    try:
        return float(m.group(0))
    except:
        return np.nan

def performance_strength(row):
    vals = []
    for m in METRIC_WEIGHTS.keys():
        if pd.notna(row.get(m)):
            vals.append(row[m] * METRIC_WEIGHTS[m])
    return np.mean(vals) if vals else np.nan

# --------------------------
# FILE UPLOAD
# --------------------------
st.title("📊 Monthly Vertical Analysis")
uploaded_file = st.file_uploader("Upload Monthly Business Review CSV", type=["csv"])

if uploaded_file:
    raw = pd.read_csv(uploaded_file, header=None)

    # --------------------------
    # PARSE BLOCKS
    # --------------------------
    tidy_rows = []
    i = 0
    while i < len(raw):
        header_cell = raw.iat[i, 0]
        if looks_like_metric_header(header_cell):
            metric = str(header_cell).strip()

            header_row = raw.iloc[i].fillna("").astype(str).str.strip().tolist()
            header_verticals = [v for v in header_row[1:] if v != ""]

            j = i + 1
            while j < len(raw) and not looks_like_metric_header(raw.iat[j, 0]):
                j += 1

            block = raw.iloc[i+1:j].copy()
            if block.empty:
                i = j
                continue

            block = block.rename(columns={block.columns[0]: "Period"})
            for k in range(1, min(len(header_verticals) + 1, block.shape[1])):
                block = block.rename(columns={block.columns[k]: header_verticals[k-1]})

            keep = ["Period"] + [v for v in VERTICALS if v in block.columns]
            block = block[keep]

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

    tidy = pd.concat(tidy_rows, ignore_index=True)

    # --------------------------
    # TREND ANALYSIS
    # --------------------------
    # Get chronological periods
    wide = tidy.pivot_table(index=["Period", "Vertical"], columns="Metric", values="Value", aggfunc="mean").reset_index()
    wide["_row_order"] = wide["Period"].rank(method="dense").astype(int)
    periods_sorted = wide.sort_values(by="_row_order")["Period"].unique().tolist()

    latest_period = periods_sorted[-1]
    prev_period = periods_sorted[-2] if len(periods_sorted) >= 2 else None

    latest_data = wide[wide["Period"] == latest_period].copy()
    if prev_period:
        prev_data = wide[wide["Period"] == prev_period].copy()
        prev_map = prev_data.set_index("Vertical")[METRIC_WEIGHTS.keys()]
        for m in METRIC_WEIGHTS.keys():
            latest_data[f"{m} Change"] = latest_data.apply(
                lambda r: r[m] - prev_map.loc[r["Vertical"], m] if r["Vertical"] in prev_map.index else np.nan, axis=1
            )

    # --------------------------
    # ZONE CLASSIFICATION
    # --------------------------
    def classify_zone(row):
        if row["Performance Strength"] >= 75:
            return "Healthy"
        elif row["Performance Strength"] >= 50:
            return "Watch"
        else:
            return "Red"

    latest_data["Performance Strength"] = latest_data.apply(performance_strength, axis=1)
    latest_data["Zone"] = latest_data.apply(classify_zone, axis=1)

    # --------------------------
    # SHOW TABLE
    # --------------------------
    st.subheader(f"📅 Latest Period: {latest_period}")
    st.dataframe(latest_data[["Vertical", "Performance Strength", "Zone"] + list(METRIC_WEIGHTS.keys())])

    # --------------------------
    # PLOTS
    # --------------------------
    metric_choice = st.selectbox("Select metric to view trends", METRIC_WEIGHTS.keys())
    fig, ax = plt.subplots(figsize=(8,5))
    for v in VERTICALS:
        subset = tidy[(tidy["Vertical"] == v) & (tidy["Metric"] == metric_choice)]
        if not subset.empty:
            ax.plot(subset["Period"], subset["Value"], marker='o', label=v)
    ax.set_title(f"Trend: {metric_choice}")
    ax.set_ylabel(metric_choice)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # --------------------------
    # REPORT GENERATION
    # --------------------------
    def generate_pdf(df):
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=landscape(A4))
        styles = getSampleStyleSheet()
        elems = []
        elems.append(Paragraph("Monthly Vertical Analysis Report", styles['Title']))
        elems.append(Spacer(1, 12))
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elems.append(table)
        doc.build(elems)
        buf.seek(0)
        return buf

    if st.button("📄 Download PDF Report"):
        pdf_buf = generate_pdf(latest_data[["Vertical", "Performance Strength", "Zone"] + list(METRIC_WEIGHTS.keys())])
        st.download_button("Download Report", pdf_buf, file_name="vertical_analysis.pdf", mime="application/pdf")

