import streamlit as st
import pandas as pd
import re
from datetime import timedelta

st.set_page_config(page_title="FTL Report Parser", layout="wide")
st.title("FTL Monthly Report Parser")

# -------------------- Helper: Time Conversion --------------------
def excel_time_to_hours(val):
    if pd.isna(val):
        return 0

    # If it's a string in format H:M or H:M:S
    if isinstance(val, str):
        val = val.strip()
        if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", val):
            tparts = [int(x) for x in val.split(":")]
            if len(tparts) == 2:
                return tparts[0] + tparts[1]/60
            elif len(tparts) == 3:
                return tparts[0] + tparts[1]/60 + tparts[2]/3600

    # If it's a timedelta
    if isinstance(val, timedelta):
        return val.total_seconds() / 3600

    # If it's a float (Excel serial time)
    try:
        return float(val) * 24
    except:
        return 0

# -------------------- Main Parser --------------------
def parse_ftl_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # Clean column headers
    df.columns = [str(c).strip() for c in df.columns]

    # Detect columns that might be durations
    time_cols = []
    for c in df.columns:
        try:
            sample = df[c].dropna().astype(str).iloc[0]
            if re.search(r"(\d{1,3}d|\d{1,2}:\d{2}(:\d{2})?)", sample) or "d" in c.lower():
                time_cols.append(c)
        except IndexError:
            continue

    # Convert and store
    for col in time_cols:
        df["hrs_" + col] = df[col].apply(excel_time_to_hours)

    return df

# -------------------- Upload & Display --------------------
uploaded = st.file_uploader("Upload FTL Monthly Report (Excel)", type=["xlsx"])
if uploaded:
    try:
        ftl = parse_ftl_excel(uploaded)
        st.success("✅ File parsed successfully!")

        # Show raw and converted hour columns
        debug_cols = [c for c in ftl.columns if any(x in c.lower() for x in ["date", "crew", "tail", "hrs_"])]
        st.dataframe(ftl[debug_cols].head(50), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")
