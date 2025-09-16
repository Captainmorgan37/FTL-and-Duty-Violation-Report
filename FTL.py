import streamlit as st
import pandas as pd
import re
from datetime import timedelta, datetime

st.set_page_config(layout="wide")
st.title("FTL Report Parser")

# ---------- Helpers ----------
def excel_time_to_hours(val):
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        # Excel serial time (fraction of a day)
        return round(float(val) * 24, 2)
    if isinstance(val, timedelta):
        return round(val.total_seconds() / 3600, 2)
    val = str(val).strip()
    try:
        # Match hh:mm:ss or hh:mm
        if re.match(r"^\d{1,2}:\d{2}(:\d{2})?$", val):
            parts = val.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s = int(parts[2]) if len(parts) == 3 else 0
            return round(h + m/60 + s/3600, 2)
        # Match X days HH:MM:SS
        days_match = re.match(r"(?:(\d+)\s+days?,?\s+)?(\d{1,2}):(\d{2})(?::(\d{2}))?", val)
        if days_match:
            d = int(days_match.group(1) or 0)
            h = int(days_match.group(2) or 0)
            m = int(days_match.group(3) or 0)
            s = int(days_match.group(4) or 0)
            return round(d * 24 + h + m/60 + s/3600, 2)
    except:
        pass
    return 0

# ---------- Main Parser ----------
def parse_ftl_excel(file):
    xl = pd.ExcelFile(file)
    sheet = xl.sheet_names[0]
    df = xl.parse(sheet)

    # Drop fully empty rows
    df.dropna(how="all", inplace=True)

    # Fix duplicate column names
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)

    # Find time-like columns by sampling first valid row
    sample_row = df.dropna(how='all').head(1)
    time_cols = []
    for col in df.columns:
        try:
            cell = str(sample_row[col].values[0])
            if re.search(r"\d{1,2}:\d{2}(:\d{2})?", cell) or 'd' in col.lower():
                time_cols.append(col)
        except:
            continue

    # Add parsed hour columns
    for col in time_cols:
        df[f"hrs_{col}"] = df[col].apply(excel_time_to_hours)

    return df

# ---------- Streamlit UI ----------
uploaded = st.file_uploader("Upload FTL Excel Report", type=["xls", "xlsx"])
if uploaded:
    try:
        ftl = parse_ftl_excel(uploaded)
        st.success("✅ FTL report parsed successfully!")

        # Show preview
        debug_cols = [c for c in ftl.columns if "hrs_" in c or any(x in c.lower() for x in ["pilot", "date", "tail", "dep", "arr"])]
        st.dataframe(ftl[debug_cols].head(30), use_container_width=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")
