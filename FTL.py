import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import re

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title="FTL & FDP Report", layout="wide")
st.title("FTL Report Parser")

# ------------------------- HELPERS -------------------------
def excel_duration_to_hours(value):
    if pd.isna(value):
        return 0.0
    if isinstance(value, str):
        try:
            tparts = [int(x) for x in value.strip().split(":")]
            if len(tparts) == 3:
                h, m, s = tparts
            elif len(tparts) == 2:
                h, m = tparts
                s = 0
            else:
                return 0.0
            return round(h + m/60 + s/3600, 2)
        except:
            return 0.0
    elif isinstance(value, (float, int)):
        return round(float(value) * 24, 2)
    elif isinstance(value, pd.Timedelta):
        return round(value.total_seconds() / 3600, 2)
    return 0.0

# ------------------------- PARSER -------------------------
def parse_ftl_excel(file):
    xl = pd.ExcelFile(file)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str)
        df["Sheet"] = sheet
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # Clean column names
    stripped_cols = [str(c).strip() for c in df.columns]
    seen = {}
    deduped_cols = []
    for col in stripped_cols:
        if col not in seen:
            seen[col] = 1
            deduped_cols.append(col)
        else:
            seen[col] += 1
            deduped_cols.append(f"{col}.{seen[col]-1}")
    df.columns = deduped_cols

    # Detect all potential time columns (string and Excel duration)
    time_cols = [c for c in df.columns if re.search(r"(\d{1,3}d|\d{1,2}:\d{2}(:\d{2})?)", str(df[c].dropna().astype(str).iloc[0])) or "d" in c.lower()]

    for col in time_cols:
        df["hrs_" + col] = df[col].apply(excel_duration_to_hours)

    return df

# ------------------------- MAIN APP -------------------------
uploaded = st.file_uploader("Upload FTL Excel Report", type=["xlsx"])
if uploaded:
    with st.spinner("Parsing report..."):
        ftl = parse_ftl_excel(uploaded)

    st.subheader("Full Parsed Data")
    st.dataframe(ftl, use_container_width=True)

    debug_cols = [c for c in ftl.columns if c.startswith("hrs_")]
    if debug_cols:
        st.subheader("Parsed Time Summary (in Hours)")
        st.dataframe(ftl[debug_cols + ["Sheet"]], use_container_width=True)
