
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO, BytesIO
from datetime import timedelta

st.set_page_config(page_title="FTL: 12+ Hour Duty Streak Checker", layout="wide")
st.title("FTL: 12+ Hour Duty Streak Checker")

st.markdown(
    "Upload the FL3XX **Flight Time Limitations (FTL)** CSV export. "
    "This app will detect **2-day** and **3-day** consecutive streaks of **12+ hour duty days** per pilot."
)

# -----------------------------
# Helpers
# -----------------------------
def parse_duration_to_hours(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan
    # Normalize various formats
    s = s.replace("hours", ":").replace("hour", ":").replace("H", ":").replace("h", ":")
    s = s.replace(" ", "")
    s = s.replace("::", ":").replace(".", ":")
    # HH:MM[:SS]
    m = re.match(r"^(\d{1,3}):(\d{1,2})(?::(\d{1,2}))?$", s)
    if m:
        h = int(m.group(1)); mi = int(m.group(2)); se = int(m.group(3)) if m.group(3) else 0
        return h + mi/60 + se/3600
    # minutes like "750m" / "750min"
    m2 = re.match(r"^(\d+)\s*(m|min)$", s, flags=re.I)
    if m2:
        return int(m2.group(1)) / 60.0
    # integer hours "12"
    if re.match(r"^\d+$", s):
        return float(int(s))
    # compound "12h30m"
    h = re.search(r"(\d+)\s*h", s, flags=re.I)
    mi = re.search(r"(\d+)\s*m", s, flags=re.I)
    if h or mi:
        hours = int(h.group(1)) if h else 0
        minutes = int(mi.group(1)) if mi else 0
        return hours + minutes/60
    # last resort: pandas timedelta
    try:
        td = pd.to_timedelta(s)
        return td.total_seconds() / 3600.0
    except Exception:
        return np.nan

def try_read_csv(uploaded_file):
    # Try flexible sep detection; fall back to semicolon
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";", engine="python", encoding="utf-8", on_bad_lines="skip")

def infer_columns(df):
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    pilot_candidates = [c for c in cols if re.search(r"(pilot|crew|user|employee|person|name)", c, re.I)]
    pilot_col = pilot_candidates[0] if pilot_candidates else None

    date_candidates = [c for c in cols if re.search(r"(date|day)", c, re.I)]
    date_col = None
    for c in date_candidates:
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.5:
            date_col = c
            break

    duty_candidates = [c for c in cols if re.search(r"(duty).*?(time|duration|hrs|hours)?", c, re.I)]
    duty_candidates += [c for c in cols if re.search(r"(total).*duty", c, re.I)]
    seen = set(); duty_candidates = [x for x in duty_candidates if not (x in seen or seen.add(x))]
    duty_col = None; best_rate = -1.0
    for c in duty_candidates:
        sample = df[c].astype(str).head(100).tolist()
        parsed = [parse_duration_to_hours(x) for x in sample]
        rate = np.mean([not pd.isna(x) for x in parsed])
        if rate > best_rate and rate > 0.3:
            best_rate = rate; duty_col = c

    if duty_col is None:
        for c in cols:
            sample = df[c].astype(str).head(200).tolist()
            parsed = [parse_duration_to_hours(x) for x in sample]
            rate = np.mean([not pd.isna(x) for x in parsed])
            plausible = [x for x in parsed if not pd.isna(x) and 1.0 <= x <= 18.0]
            if rate > 0.4 and len(plausible) >= 10:
                duty_col = c; break

    return pilot_col, date_col, duty_col

def build_work_table(df, pilot_col, date_col, duty_col, dayfirst=True, min_hours=12.0):
    # Forward-fill pilot names through blank rows in their block
    df[pilot_col] = df[pilot_col].ffill()

    work = df[[pilot_col, date_col, duty_col]].copy()
    work.columns = ["Pilot", "Date", "DutyRaw"]
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce", dayfirst=dayfirst)
    work["DutyHours"] = work["DutyRaw"].map(parse_duration_to_hours)

    # Keep rows that have a date & duty value
    work = work.dropna(subset=["Pilot", "Date", "DutyHours"])

    # If multiple entries per pilot/day, keep the max duty that day
    work = work.sort_values(["Pilot", "Date"]).groupby(["Pilot", "Date"], as_index=False)["DutyHours"].max()
    work["Long"] = work["DutyHours"] >= float(min_hours)
    return work

def streaks(work, min_consecutive=3):
    sequences = []
    for pilot, sub in work.groupby("Pilot"):
        sub = sub.sort_values("Date").reset_index(drop=True)
        count = 0; start_date = None; last_long_date = None
        for _, r in sub.iterrows():
            if r["Long"]:
                if last_long_date is not None and r["Date"] == last_long_date + timedelta(days=1):
                    count += 1
                else:
                    count = 1
                    start_date = r["Date"]
                last_long_date = r["Date"]
            else:
                if count >= min_consecutive and start_date is not None:
                    sequences.append({
                        "Pilot": pilot,
                        "StartDate": start_date.date().isoformat(),
                        "EndDate": last_long_date.date().isoformat(),
                        "ConsecutiveDays": count
                    })
                count = 0; start_date = None; last_long_date = None
        if count >= min_consecutive and start_date is not None:
            sequences.append({
                "Pilot": pilot,
                "StartDate": start_date.date().isoformat(),
                "EndDate": last_long_date.date().isoformat(),
                "ConsecutiveDays": count
            })
    return pd.DataFrame(sequences).sort_values(["Pilot", "StartDate"]) if sequences else pd.DataFrame(
        columns=["Pilot", "StartDate", "EndDate", "ConsecutiveDays"]
    )

def coverage_table(work):
    cov = work.groupby("Pilot").agg(
        FirstDate=("Date", "min"),
        LastDate=("Date", "max"),
        DaysCount=("Date", "nunique"),
        LongDays=("Long", "sum")
    ).reset_index().sort_values("Pilot")
    cov["FirstDate"] = cov["FirstDate"].dt.date
    cov["LastDate"] = cov["LastDate"].dt.date
    return cov

def to_csv_download(df, filename):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return st.download_button("Download " + filename, data=csv_bytes, file_name=filename, mime="text/csv")

# -----------------------------
# Sidebar options
# -----------------------------
with st.sidebar:
    st.header("Options")
    dayfirst = st.checkbox("Day-first dates (DD/MM/YYYY)", value=True)
    min_hours = st.number_input("Minimum hours to count as 'long day'", min_value=1.0, max_value=24.0, value=12.0, step=0.5)
    st.caption("Defaults match your 12+ hour rule; you can adjust if needed.")

uploaded = st.file_uploader("Upload FL3XX FTL CSV", type=["csv"])

if uploaded is not None:
    # Read
    df = try_read_csv(uploaded)
    pilot_col, date_col, duty_col = infer_columns(df)

    st.subheader("Column Mapping")
    cols = list(df.columns)

    pilot_col = st.selectbox("Pilot column", options=cols, index=cols.index(pilot_col) if pilot_col in cols else 0)
    date_col = st.selectbox("Date column", options=cols, index=cols.index(date_col) if date_col in cols else 0)
    duty_col = st.selectbox("Duty duration column", options=cols, index=cols.index(duty_col) if duty_col in cols else 0)

    if st.button("Run Check", type="primary"):
        # Build normalized duty-by-day
        work = build_work_table(df.copy(), pilot_col, date_col, duty_col, dayfirst=dayfirst, min_hours=min_hours)
        cov = coverage_table(work)

        st.markdown("### Per-Pilot Coverage")
        st.dataframe(cov, use_container_width=True)
        to_csv_download(cov, "FTL_Per_Pilot_Coverage.csv")

        # Streaks for 2 and 3 consecutive days
        st.markdown("### Streaks (≥ 2 consecutive long days)")
        seq2 = streaks(work, min_consecutive=2)
        st.dataframe(seq2, use_container_width=True)
        to_csv_download(seq2, "FTL_2xLong_Consecutive_Summary.csv")

        st.markdown("### Streaks (≥ 3 consecutive long days)")
        seq3 = streaks(work, min_consecutive=3)
        st.dataframe(seq3, use_container_width=True)
        to_csv_download(seq3, "FTL_3xLong_Consecutive_Summary.csv")

        # Full normalized detail
        st.markdown("### Parsed Duty by Day (normalized)")
        st.dataframe(work, use_container_width=True)
        to_csv_download(work, "FTL_Duty_By_Day_Parsed.csv")

else:
    st.info("Upload the FTL CSV to begin.")

st.markdown("---")
st.caption("Tip: If your export lists a pilot name once with blank rows beneath, this tool forward-fills the pilot name to those rows automatically.")
