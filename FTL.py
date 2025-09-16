import re
import math
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="FTL/FRMS Checker", layout="wide")
st.title("AirSprint FRMS & FTL Duty Checker")

# ---------- Helpers ----------
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")

def parse_hhmm_from_text(val):
    if pd.isna(val): return None
    s = str(val); m = TIME_RE.search(s)
    if not m: return None
    hh, mm = int(m.group(1)), int(m.group(2))
    try:
        return datetime.strptime(f"{hh:02d}:{mm:02d}", "%H:%M").time()
    except:
        return None

def duration_to_minutes(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return None
    s = str(val).strip()
    if s == "" or s.lower() == "none": return None
    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0]); m = int(parts[1]) if len(parts) > 1 else 0
            sec = int(parts[2]) if len(parts) > 2 else 0
        except: return None
        return h*60 + m + sec/60.0
    try: return float(s)*60.0
    except: return None

def minutes_to_hours(mins):
    if mins is None or (isinstance(mins, float) and pd.isna(mins)): return None
    return round(mins/60.0, 2)

def to_dt(d, t):
    return None if pd.isna(d) or t is None else datetime.combine(d, t)

# ---------- FDP consolidation ----------
def consolidate_fdps(ftl):
    ftl = ftl.sort_values(["Name", "Date_parsed", "RowOrder"]).reset_index(drop=True)
    periods = []
    cur = None

    for _, row in ftl.iterrows():
        name = row["Name"]
        date = row["Date_parsed"]

        sd_raw = str(row.get("Start Duty", "") or "")
        duty_raw = str(row.get("Duty", "") or "")

        start_t = row["StartDuty_t"]
        end_t = row["BlocksOn_t"]

        is_split = "(split)" in sd_raw.lower()
        is_positioning = duty_raw.strip().startswith("P ")
        is_sim_evt = duty_raw.strip().startswith(("SIM", "EVT"))
        is_new_start = start_t is not None and not (is_positioning or is_sim_evt)

        if cur is None or name != cur["Name"]:
            if cur is not None:
                periods.append(cur)
            cur = {
                "Name": name,
                "Date": date,
                "duty_start": start_t,
                "fdp_end": end_t if (end_t and not (is_positioning or is_sim_evt)) else None,
                "duty_end": end_t,
                "hrs7d": row["hrs7d"],
                "hrs30d": row["hrs30d"],
                "split": False,
                "break_min": None,
            }
            continue

        if is_split or is_new_start:
            periods.append(cur)
            cur = {
                "Name": name,
                "Date": date if date else cur["Date"],   # inherit date if missing
                "duty_start": start_t if start_t else cur["duty_start"],
                "fdp_end": end_t if (end_t and not (is_positioning or is_sim_evt)) else None,
                "duty_end": end_t if end_t else cur["duty_end"],
                "hrs7d": row["hrs7d"],
                "hrs30d": row["hrs30d"],
                "split": is_split,
                "break_min": None,
            }
            continue

        if end_t:
            cur["duty_end"] = end_t
            if not (is_positioning or is_sim_evt):
                cur["fdp_end"] = end_t

        if pd.notna(row["hrs7d"]):
            cur["hrs7d"] = row["hrs7d"]
        if pd.notna(row["hrs30d"]):
            cur["hrs30d"] = row["hrs30d"]

        if not date and cur["Date"]:
            row["Date"] = cur["Date"]

    if cur is not None:
        periods.append(cur)

    return pd.DataFrame(periods)

# ---------- File upload ----------
uploaded = st.file_uploader("Upload FL3XX FTL CSV", type=["csv"])

if uploaded:
    try:
        ftl = pd.read_csv(uploaded, engine="python")
    except:
        ftl = pd.read_csv(uploaded)

    ftl["Name"] = ftl["Name"].ffill()
    ftl["RowOrder"] = range(len(ftl))
    ftl["Date_parsed"] = pd.to_datetime(ftl["Date"], errors="coerce").dt.date

    ftl["StartDuty_t"] = ftl["Start Duty"].apply(parse_hhmm_from_text)
    ftl["BlocksOn_t"] = ftl["Blocks On"].apply(parse_hhmm_from_text)

    ftl["hrs7d"] = ftl["7d"].apply(duration_to_minutes).apply(minutes_to_hours) if "7d" in ftl.columns else pd.NA
    ftl["hrs30d"] = ftl["30d"].apply(duration_to_minutes).apply(minutes_to_hours) if "30d" in ftl.columns else pd.NA

    fdp = consolidate_fdps(ftl)

    # FDP + Duty length
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
            if fdp_end < s:
                fdp_end += timedelta(days=1)
            fdp.at[i, "FDP_min"] = (fdp_end - s).total_seconds() / 60.0

        if duty_e < s:
            duty_e += timedelta(days=1)
        fdp.at[i, "Duty_min"] = (duty_e - s).total_seconds() / 60.0

    fdp["FDP_hrs"] = fdp["FDP_min"].apply(minutes_to_hours)
    fdp["Duty_hrs"] = fdp["Duty_min"].apply(minutes_to_hours)

    # Turns
    fdp = fdp.sort_values(["Name", "Date", "duty_start", "duty_end"]).reset_index(drop=True)
    fdp["Turn_min"] = None

    for i in range(1, len(fdp)):
        if fdp.loc[i, "Name"] == fdp.loc[i-1, "Name"]:
            if not fdp.loc[i, "split"]:
                prev_end = to_dt(fdp.loc[i-1, "Date"], fdp.loc[i-1, "duty_end"])
                cur_start = to_dt(fdp.loc[i, "Date"], fdp.loc[i, "duty_start"])
                if prev_end and cur_start:
                    if fdp.loc[i-1, "fdp_end"]:  # only add +15 if last duty was flown
                        prev_end += timedelta(minutes=15)
                    if cur_start < prev_end:
                        cur_start += timedelta(days=1)
                    fdp.at[i, "Turn_min"] = (cur_start - prev_end).total_seconds() / 60.0

    fdp["Turn_hrs"] = fdp["Turn_min"].apply(minutes_to_hours)

    # ---------- Rule checks ----------
    issues = []
    def add_issue(name, date, rule, details):
        issues.append({"Name": name, "Date": date, "Rule": rule, "Details": details})

    for i, r in fdp.iterrows():
        fdp_min = r["FDP_min"]
        duty_min = r["Duty_min"]
        turn_min = r["Turn_min"]

        if fdp_min and fdp_min > 15*60:
            add_issue(r["Name"], r["Date"], "FDP >15h",
                      f"FDP {minutes_to_hours(fdp_min)}h")

        if duty_min and duty_min > 14*60:
            excess = duty_min - 14*60
            required_rest = 10*60 + math.ceil(excess/2.0)
            if i+1 < len(fdp) and fdp.loc[i+1, "Name"] == r["Name"]:
                next_turn = fdp.loc[i+1, "Turn_min"]
                if next_turn and next_turn < required_rest:
                    add_issue(r["Name"], r["Date"], "Post-duty rest too short",
                              f"Duty {minutes_to_hours(duty_min)}h exceeded 14h by {minutes_to_hours(excess)}h → rest must be ≥{minutes_to_hours(required_rest)}h")

        if turn_min and turn_min < 10*60:
            add_issue(r["Name"], r["Date"], "Turn <10h",
                      f"Turn {minutes_to_hours(turn_min)}h")

    issues_df = pd.DataFrame(issues)

    # ---------- UI ----------
    st.subheader("Consolidated FDPs (debug)")
    dbg_cols = ["Name", "Date", "duty_start", "fdp_end", "duty_end", "FDP_hrs", "Duty_hrs", "Turn_hrs", "hrs7d", "hrs30d"]
    st.dataframe(fdp[dbg_cols], use_container_width=True)

    st.subheader("Exceedances Detected")
    if issues_df.empty:
        st.success("No exceedances found ✅")
    else:
        st.error(f"{len(issues_df)} exceedance(s) found")
        st.dataframe(issues_df, use_container_width=True)
        st.download_button("Download exceedances (CSV)",
                           issues_df.to_csv(index=False),
                           file_name="exceedances.csv",
                           mime="text/csv")
