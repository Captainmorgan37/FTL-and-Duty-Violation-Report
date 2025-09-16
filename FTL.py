import re
import math
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, time

st.set_page_config(page_title="FTL/FRMS Checker", layout="wide")
st.title("AirSprint FRMS & FTL Duty Checker")

# ---------- Helpers ----------
def excel_time_to_time(val):
    """Convert Excel-style hh:mm:ss into datetime.time"""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        if val.year == 1900:  # Excel dummy anchor for times
            return time(val.hour, val.minute, val.second)
        return val.time()
    if isinstance(val, pd.Timedelta):
        total_seconds = int(val.total_seconds())
        h, m = divmod(total_seconds // 60, 60)
        s = total_seconds % 60
        return time(h % 24, m, s)
    s = str(val).strip()
    if ":" in s:
        try:
            parts = [int(x) for x in s.split(":")]
            while len(parts) < 3:
                parts.append(0)
            return time(parts[0], parts[1], parts[2])
        except:
            return None
    return None

def excel_time_to_hours(val):
    """Convert Excel-style hh:mm:ss, Timedelta, or 1900-anchored datetime-with-days into float hours"""
    if pd.isna(val):
        return None

    # Case 1: Excel serial datetime anchored near 1900-01-01
    if isinstance(val, pd.Timestamp):
        anchor = pd.Timestamp(1900, 1, 1)
        try:
            days = (val.normalize() - anchor).days
            if days < 0 or days > 4000:  # If it's clearly a real date, ignore days
                days = 0
        except Exception:
            days = 0
        return days * 24.0 + val.hour + val.minute/60.0 + val.second/3600.0

    # Case 2: True pandas Timedelta
    if isinstance(val, pd.Timedelta):
        return val.total_seconds() / 3600.0

    # Case 3: String like "7 days 08:20:00" or "2 days"
    s = str(val).strip().lower()
    if "day" in s:
        parts = s.replace("days", "day").split("day")
        try:
            days = int(parts[0].strip())
        except:
            days = 0
        hh = mm = ss = 0
        tail = parts[1].strip() if len(parts) > 1 else ""
        if ":" in tail:
            tparts = [int(x) for x in tail.split(":")]
            while len(tparts) < 3:
                tparts.append(0)
            hh, mm, ss = tparts[:3]
        return days * 24.0 + hh + mm/60.0 + ss/3600.0

    if ":" in s:
        try:
            parts = [int(x) for x in s.split(":")]
            while len(parts) < 3:
                parts.append(0)
            hh, mm, ss = parts[:3]
            return hh + mm/60.0 + ss/3600.0
        except:
            return None

    try:
        return float(s)
    except:
        return None

def to_dt(d, t):
    return None if pd.isna(d) or t is None else datetime.combine(d, t)

# ---------- Parser ----------
def parse_ftl_excel(file, sheet=0):
    df = pd.read_excel(file, sheet_name=sheet, engine="openpyxl")

    # Forward-fill pilot name
    if "Name" in df.columns:
        df["Name"] = df["Name"].ffill()

    # Parse date column
    if "Date" in df.columns:
        df["Date_parsed"] = pd.to_datetime(
            df["Date"], format="%d.%m.%Y", errors="coerce"
        ).dt.date

    # Auto-detect time-of-day columns
    time_like_cols = []
    for col in df.columns:
        if col in ["Date", "Name"]:
            continue
        sample = df[col].dropna().head(10)
        if not sample.empty:
            if any(isinstance(v, (pd.Timestamp, pd.Timedelta)) for v in sample):
                time_like_cols.append(col)
            elif any(":" in str(v) for v in sample.astype(str)):
                time_like_cols.append(col)

    for col in time_like_cols:
        df[col + "_t"] = df[col].apply(excel_time_to_time)

    # Auto-detect rolling/duration columns
    duration_keys = ["7d", "28d", "30d", "90d", "180d", "365d"]
    duration_cols = [c for c in df.columns if any(key in c.lower() for key in duration_keys)]
    duration_cols += [c for c in df.columns if pd.api.types.is_timedelta64_ns_dtype(df[c])]
    seen = set()
    duration_cols = [c for c in duration_cols if not (c in seen or seen.add(c))]

    for col in duration_cols:
        df["hrs_" + col] = df[col].apply(excel_time_to_hours)

    return df

# ---------- FDP consolidation ----------
def consolidate_fdps(ftl):
    ftl = ftl.sort_values(["Name", "Date_parsed"]).reset_index(drop=True)
    periods = []
    cur = None

    for _, row in ftl.iterrows():
        name = row["Name"]
        date = row["Date_parsed"]
        start_t = row.get("Start Duty_t")
        end_t = row.get("Blocks On_t")

        duty_raw = str(row.get("Duty", "") or "")
        reason_raw = str(row.get("Reason", "") or "")
        is_positioning = duty_raw.strip().startswith("P ")
        is_sim_evt = duty_raw.strip().startswith(("SIM", "EVT"))
        is_rest = "rest" in reason_raw.lower()

        this_date = date if pd.notna(date) else (cur["Date"] if cur else None)

        # Reset rows
        if is_rest or is_positioning or is_sim_evt or (pd.notna(date) and start_t is None and end_t is None):
            if cur is not None:
                periods.append(cur)
            periods.append({
                "Name": name,
                "Date": this_date,
                "duty_start": None,
                "fdp_end": None,
                "duty_end": None,
                "hrs7d": row.get("hrs_7d"),
                "hrs28d": row.get("hrs_28d"),
                "hrs30d": row.get("hrs_30d"),
                "hrs365d": row.get("hrs_365d"),
            })
            cur = None
            continue

        # New pilot or first duty
        if cur is None or name != cur["Name"]:
            if cur is not None:
                periods.append(cur)
            cur = {
                "Name": name,
                "Date": this_date,
                "duty_start": start_t,
                "fdp_end": end_t,
                "duty_end": end_t,
                "hrs7d": row.get("hrs_7d"),
                "hrs28d": row.get("hrs_28d"),
                "hrs30d": row.get("hrs_30d"),
                "hrs365d": row.get("hrs_365d"),
            }
            continue

        # Extend current duty
        if end_t:
            cur["duty_end"] = end_t
            cur["fdp_end"] = end_t

        if pd.notna(row.get("hrs_7d")):
            cur["hrs7d"] = row["hrs_7d"]
        if pd.notna(row.get("hrs_28d")):
            cur["hrs28d"] = row["hrs_28d"]
        if pd.notna(row.get("hrs_30d")):
            cur["hrs30d"] = row["hrs_30d"]
        if pd.notna(row.get("hrs_365d")):
            cur["hrs365d"] = row["hrs_365d"]

        if this_date:
            cur["Date"] = this_date

    if cur is not None:
        periods.append(cur)

    return pd.DataFrame(periods)

# ---------- File upload ----------
uploaded = st.file_uploader("Upload FTL Excel Report", type=["xlsx", "csv"])

if uploaded:
    ftl = parse_ftl_excel(uploaded)

    # Debug view: only show parsed versions (_t and hrs_)
    debug_cols = ["Name", "Date", "Date_parsed"]
    debug_cols += [c for c in ftl.columns if c.endswith("_t")]       # parsed times
    debug_cols += [c for c in ftl.columns if c.startswith("hrs_")]   # parsed hours
    st.subheader("Parsed Data (debug)")
    st.dataframe(ftl[debug_cols].head(30), use_container_width=True)

    fdp = consolidate_fdps(ftl)

    # FDP + Duty mins
    fdp["FDP_min"] = None
    fdp["Duty_min"] = None

    for i, r in fdp.iterrows():
        s = to_dt(r["Date"], r["duty_start"])
        fdp_e = to_dt(r["Date"], r["fdp_end"]) if r["fdp_end"] else None
        duty_e = to_dt(r["Date"], r["duty_end"]) if r["duty_end"] else None

        if not s or not duty_e:
            continue

        if fdp_e:
            fdp_end = fdp_e + timedelta(minutes=15)
            if fdp_end < s or (s.hour > 18 and fdp_end.hour < 6):
                fdp_end += timedelta(days=1)
            fdp.at[i, "FDP_min"] = (fdp_end - s).total_seconds() / 60.0

        if duty_e < s or (s.hour > 18 and duty_e.hour < 6):
            duty_e += timedelta(days=1)
        fdp.at[i, "Duty_min"] = (duty_e - s).total_seconds() / 60.0

    fdp["FDP_hrs"] = fdp["FDP_min"].apply(lambda x: round(x/60, 2) if x else None)
    fdp["Duty_hrs"] = fdp["Duty_min"].apply(lambda x: round(x/60, 2) if x else None)

    st.subheader("Consolidated FDPs (debug)")
    dbg_cols = ["Name", "Date", "duty_start", "fdp_end", "duty_end",
                "FDP_hrs", "Duty_hrs", "hrs7d", "hrs28d", "hrs30d", "hrs365d"]
    st.dataframe(fdp[dbg_cols], use_container_width=True)
