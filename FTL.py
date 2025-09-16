
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta

st.set_page_config(page_title="FTL Audit: Duty & Rest Checks (Locked)", layout="wide")
st.title("FTL Audit: Duty & Rest Checks (Locked)")

st.markdown(
    "Upload the FL3XX **Flight Time Limitations (FTL)** CSV export. "
    "This runs two locked checks per pilot:"
    "\n\n- **Duty Streaks**: ≥2 and ≥3 consecutive **12+ hr duty** days"
    "\n- **Short Rest**: ≥2 consecutive days with **rest < 11 hours**"
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
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";", engine="python", encoding="utf-8", on_bad_lines="skip")

def infer_common_columns(df):
    # Normalize headers
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

    return pilot_col, date_col

def infer_duty_column(df):
    cols = list(df.columns)
    duty_candidates = [c for c in cols if re.search(r"(duty).*?(time|duration|hrs|hours)?", c, re.I)]
    duty_candidates += [c for c in cols if re.search(r"(total).*duty", c, re.I)]
    seen = set(); duty_candidates = [x for x in duty_candidates if not (x in seen or seen.add(x))]
    duty_col = None; best_rate = -1.0
    for c in duty_candidates:
        sample = df[c].astype(str).head(300).tolist()
        parsed = [parse_duration_to_hours(x) for x in sample]
        rate = np.mean([not pd.isna(x) for x in parsed])
        if rate > best_rate and rate > 0.3:
            best_rate = rate; duty_col = c
    if duty_col is None:
        for c in cols:
            sample = df[c].astype(str).head(600).tolist()
            parsed = [parse_duration_to_hours(x) for x in sample]
            rate = np.mean([not pd.isna(x) for x in parsed])
            plausible = [x for x in parsed if not pd.isna(x) and 1.0 <= x <= 18.0]
            if rate > 0.4 and len(plausible) >= 10:
                duty_col = c; break
    return duty_col

def infer_rest_column(df):
    cols = list(df.columns)
    rest_candidates = []
    rest_candidates += [c for c in cols if re.search(r"\brest\b", c, re.I)]
    rest_candidates += [c for c in cols if re.search(r"(assumed|deemed).*\brest\b", c, re.I)]
    rest_candidates += [c for c in cols if re.search(r"\b(min|minimum)\s*rest\b", c, re.I)]
    rest_candidates += [c for c in cols if re.search(r"rest.*(time|duration|hrs|hours)", c, re.I)]
    seen = set(); rest_candidates = [x for x in rest_candidates if not (x in seen or seen.add(x))]
    rest_col = None; best_rate = -1.0
    for c in rest_candidates:
        sample = df[c].astype(str).head(300).tolist()
        parsed = [parse_duration_to_hours(x) for x in sample]
        rate = np.mean([not pd.isna(x) for x in parsed])
        plausible = [x for x in parsed if not pd.isna(x) and 0 < x <= 48.0]
        if rate > best_rate and (rate > 0.25 or len(plausible) >= 10):
            best_rate = rate; rest_col = c
    if rest_col is None:
        for c in cols:
            sample = df[c].astype(str).head(600).tolist()
            parsed = pd.Series([parse_duration_to_hours(x) for x in sample])
            if parsed.notna().mean() > 0.4:
                med = parsed.dropna()
                if not med.empty and 4 <= med.median() <= 24:
                    rest_col = c
                    break
    return rest_col

def build_duty_table(df, pilot_col, date_col, duty_col, min_hours=12.0):
    df[pilot_col] = df[pilot_col].ffill()
    work = df[[pilot_col, date_col, duty_col]].copy()
    work.columns = ["Pilot", "Date", "DutyRaw"]
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce", dayfirst=True)
    work["DutyHours"] = work["DutyRaw"].map(parse_duration_to_hours)
    work = work.dropna(subset=["Pilot", "Date", "DutyHours"])
    work = work.sort_values(["Pilot", "Date"]).groupby(["Pilot", "Date"], as_index=False)["DutyHours"].max()
    work["LongDuty"] = work["DutyHours"] >= float(min_hours)
    return work

def build_rest_table(df, pilot_col, date_col, rest_col, short_thresh=11.0):
    df[pilot_col] = df[pilot_col].ffill()
    rest = df[[pilot_col, date_col, rest_col]].copy()
    rest.columns = ["Pilot", "Date", "RestRaw"]
    rest["Date"] = pd.to_datetime(rest["Date"], errors="coerce", dayfirst=True)
    rest["RestHours"] = rest["RestRaw"].map(parse_duration_to_hours)
    rest = rest.dropna(subset=["Pilot", "Date", "RestHours"])
    # per day, choose MIN rest observed (conservative)
    rest = rest.sort_values(["Pilot", "Date"]).groupby(["Pilot", "Date"], as_index=False)["RestHours"].min()
    rest["ShortRest"] = rest["RestHours"] < float(short_thresh)
    return rest

def streaks(df, flag_col, min_consecutive=3):
    sequences = []
    for pilot, sub in df.groupby("Pilot"):
        sub = sub.sort_values("Date").reset_index(drop=True)
        count = 0; start_date = None; last_date = None
        for _, r in sub.iterrows():
            if r[flag_col]:
                if last_date is not None and r["Date"] == last_date + timedelta(days=1):
                    count += 1
                else:
                    count = 1
                    start_date = r["Date"]
                last_date = r["Date"]
            else:
                if count >= min_consecutive and start_date is not None:
                    sequences.append({
                        "Pilot": pilot,
                        "StartDate": start_date.date().isoformat(),
                        "EndDate": last_date.date().isoformat(),
                        "ConsecutiveDays": count
                    })
                count = 0; start_date = None; last_date = None
        if count >= min_consecutive and start_date is not None:
            sequences.append({
                "Pilot": pilot,
                "StartDate": start_date.date().isoformat(),
                "EndDate": last_date.date().isoformat(),
                "ConsecutiveDays": count
            })
    return pd.DataFrame(sequences).sort_values(["Pilot", "StartDate"]) if sequences else pd.DataFrame(
        columns=["Pilot", "StartDate", "EndDate", "ConsecutiveDays"]
    )

def coverage_table(df, flag_col_name):
    cov = df.groupby("Pilot").agg(
        FirstDate=("Date", "min"),
        LastDate=("Date", "max"),
        DaysCount=("Date", "nunique"),
        FlaggedDays=(flag_col_name, "sum")
    ).reset_index().sort_values("Pilot")
    cov["FirstDate"] = cov["FirstDate"].dt.date
    cov["LastDate"] = cov["LastDate"].dt.date
    return cov

def to_csv_download(df, filename, key=None):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download " + filename, data=csv_bytes, file_name=filename, mime="text/csv", key=key)

# -----------------------------
# Main
# -----------------------------
uploaded = st.file_uploader("Upload FL3XX FTL CSV", type=["csv"])

if uploaded is not None:
    df = try_read_csv(uploaded)
    pilot_col, date_col = infer_common_columns(df.copy())

    if not pilot_col or not date_col:
        st.error("Could not confidently identify common columns (Pilot, Date).")
        diag = pd.DataFrame([{
            "pilot_col": pilot_col,
            "date_col": date_col,
            "columns_found": ", ".join(list(df.columns)[:60]) + ("..." if len(df.columns) > 60 else "")
        }])
        st.dataframe(diag, use_container_width=True)
    else:
        duty_col = infer_duty_column(df.copy())
        rest_col = infer_rest_column(df.copy())

        if not duty_col and not rest_col:
            st.error("Could not identify Duty or Rest columns. Please review your export.")
            st.write("Columns found:", list(df.columns)[:60])
        else:
            # Build tables as available
            duty_work = None; rest_work = None
            if duty_col:
                duty_work = build_duty_table(df.copy(), pilot_col, date_col, duty_col, min_hours=12.0)
                seq2_duty = streaks(duty_work, "LongDuty", min_consecutive=2)
                seq3_duty = streaks(duty_work, "LongDuty", min_consecutive=3)
            if rest_col:
                rest_work = build_rest_table(df.copy(), pilot_col, date_col, rest_col, short_thresh=11.0)
                seq2_rest = streaks(rest_work, "ShortRest", min_consecutive=2)

            results_tab, debug_tab = st.tabs(["Results", "Debug"])

            with results_tab:
                if duty_work is not None:
                    st.subheader("Duty Streaks (12+ hr days)")
                    if duty_work is not None and not seq3_duty.empty:
                        pilots = sorted(seq3_duty["Pilot"].unique().tolist())
                        st.error(f"⚠️ 3-day duty rule TRIGGERED: {len(pilots)} pilot(s): {', '.join(pilots)}")
                    else:
                        st.success("✅ No pilots with ≥3 consecutive 12+ hr duty days detected.")

                    st.markdown("**≥ 3 consecutive 12+ hr duty days**")
                    st.dataframe(seq3_duty if duty_work is not None else pd.DataFrame(), use_container_width=True)
                    if duty_work is not None:
                        to_csv_download(seq3_duty, "FTL_3x12hr_Consecutive_Duty_Summary.csv", key="dl_duty3")

                    st.markdown("**≥ 2 consecutive 12+ hr duty days**")
                    st.dataframe(seq2_duty if duty_work is not None else pd.DataFrame(), use_container_width=True)
                    if duty_work is not None:
                        to_csv_download(seq2_duty, "FTL_2x12hr_Consecutive_Duty_Summary.csv", key="dl_duty2")

                st.markdown("---")

                if rest_work is not None:
                    st.subheader("Short Rest (< 11 hr)")
                    if not seq2_rest.empty:
                        pilots_r = sorted(seq2_rest["Pilot"].unique().tolist())
                        st.error(f"⚠️ Short Rest TRIGGERED: {len(pilots_r)} pilot(s) with ≥2 consecutive days of rest < 11 hr: {', '.join(pilots_r)}")
                    else:
                        st.success("✅ No pilots with ≥2 consecutive days of rest < 11 hr detected.")

                    st.markdown("**≥ 2 consecutive days with rest < 11 hr**")
                    st.dataframe(seq2_rest, use_container_width=True)
                    to_csv_download(seq2_rest, "FTL_Consecutive_Short_Rest_Summary.csv", key="dl_rest2")

            with debug_tab:
                st.caption("Diagnostics & normalized tables (for troubleshooting)")

                if duty_work is not None:
                    st.markdown("### Duty — Per-Pilot Coverage")
                    cov_duty = coverage_table(duty_work, "LongDuty")
                    st.dataframe(cov_duty, use_container_width=True)
                    to_csv_download(cov_duty, "FTL_Duty_Per_Pilot_Coverage.csv", key="dl_cov_duty")

                    st.markdown("### Duty — Parsed Duty by Day (normalized)")
                    st.dataframe(duty_work, use_container_width=True)
                    to_csv_download(duty_work, "FTL_Duty_By_Day_Parsed.csv", key="dl_work_duty")

                if rest_work is not None:
                    st.markdown("### Rest — Per-Pilot Coverage")
                    cov_rest = coverage_table(rest_work, "ShortRest")
                    st.dataframe(cov_rest, use_container_width=True)
                    to_csv_download(cov_rest, "FTL_Rest_Per_Pilot_Coverage.csv", key="dl_cov_rest")

                    st.markdown("### Rest — Parsed Rest by Day (normalized)")
                    st.dataframe(rest_work, use_container_width=True)
                    to_csv_download(rest_work, "FTL_Rest_By_Day_Parsed.csv", key="dl_work_rest")
else:
    st.info("Upload the FTL CSV to begin.")

st.markdown("---")
st.caption("Locked inference: Pilot and Date detected, Pilot names forward-filled. Dates parsed as day-first (DD/MM/YYYY).")
