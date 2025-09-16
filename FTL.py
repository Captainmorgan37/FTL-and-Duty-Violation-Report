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

def parse_date_cell(val):
    """Parse dates like '12.08.2025' (dd.mm.yyyy) or other strings reliably."""
    if pd.isna(val): 
        return pd.NaT
    s = str(val).strip()
    if not s:
        return pd.NaT
    # Exact dd.mm.yyyy (with leading zeros) — most FL3XX exports look like this
    try:
        if "." in s:
            return datetime.strptime(s, "%d.%m.%Y").date()
    except:
        pass
    # Fallback: let pandas parse but prefer dayfirst semantics
    try:
        return pd.to_datetime(s, dayfirst=True, errors="raise").date()
    except:
        return pd.NaT

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
        reason_raw = str(row.get("Reason", "") or "")

        start_t = row["StartDuty_t"]
        end_t = row["BlocksOn_t"]

        is_split = "(split)" in sd_raw.lower()
        is_positioning = duty_raw.strip().startswith("P ")
        is_sim_evt = duty_raw.strip().startswith(("SIM", "EVT"))
        is_rest = "rest" in reason_raw.lower()

        # Always carry a date forward
        this_date = date if pd.notna(date) else (cur["Date"] if cur else None)

        # --- Explicit reset rows (Rest, SIM, Positioning, or date-only marker) ---
        if is_rest or is_positioning or is_sim_evt or (pd.notna(date) and start_t is None and end_t is None):
            if cur is not None:
                periods.append(cur)
            # Keep a placeholder so the date isn't dropped
            periods.append({
                "Name": name,
                "Date": this_date,
                "duty_start": None,
                "fdp_end": None,
                "duty_end": None,
                "hrs7d": row["hrs7d"],
                "hrs30d": row["hrs30d"],
                "split": False,
                "break_min": None,
            })
            cur = None
            continue

        # --- New pilot or first duty ---
        if cur is None or name != cur["Name"]:
            if cur is not None:
                periods.append(cur)
            cur = {
                "Name": name,
                "Date": this_date,
                "duty_start": start_t,
                "fdp_end": end_t,
                "duty_end": end_t,
                "hrs7d": row["hrs7d"],
                "hrs30d": row["hrs30d"],
                "split": is_split,
                "break_min": None,
            }
            continue

        # --- Split duty starts new FDP ---
        if is_split:
            periods.append(cur)
            cur = {
                "Name": name,
                "Date": this_date,
                "duty_start": start_t if start_t else cur["duty_start"],
                "fdp_end": end_t,
                "duty_end": end_t,
                "hrs7d": row["hrs7d"],
                "hrs30d": row["hrs30d"],
                "split": True,
                "break_min": None,
            }
            continue

        # --- Extend current duty ---
        if end_t:
            cur["duty_end"] = end_t
            cur["fdp_end"] = end_t

        if pd.notna(row["hrs7d"]):
            cur["hrs7d"] = row["hrs7d"]
        if pd.notna(row["hrs30d"]):
            cur["hrs30d"] = row["hrs30d"]

        if this_date:
            cur["Date"] = this_date

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

    # Carry pilot name through blank cells
    ftl["Name"] = ftl["Name"].ffill()
    ftl["RowOrder"] = range(len(ftl))

    # Robust date parsing + per-pilot forward-fill
    ftl["Date_parsed"] = ftl["Date"].apply(parse_date_cell)
    ftl["Date_parsed"] = ftl.groupby("Name")["Date_parsed"].ffill()

    # Times
    ftl["StartDuty_t"] = ftl["Start Duty"].apply(parse_hhmm_from_text)
    ftl["BlocksOn_t"] = ftl["Blocks On"].apply(parse_hhmm_from_text)

    # Rolling hours (converted to hours float)
    ftl["hrs7d"] = ftl["7d"].apply(duration_to_minutes).apply(minutes_to_hours) if "7d" in ftl.columns else pd.NA
    ftl["hrs30d"] = ftl["30d"].apply(duration_to_minutes).apply(minutes_to_hours) if "30d" in ftl.columns else pd.NA

    fdp = consolidate_fdps(ftl)

    # FDP + Duty minutes
    fdp["FDP_min"] = None
    fdp["Duty_min"] = None

    for i, r in fdp.iterrows():
        s = to_dt(r["Date"], r["duty_start"])
        fdp_e = to_dt(r["Date"], r["fdp_end"]) if r["fdp_end"] else None
        duty_e = to_dt(r["Date"], r["duty_end"]) if r["duty_end"] else None

        if not s or not duty_e:
            continue

        # FDP end (+15 min) with post-midnight roll
        if fdp_e:
            fdp_end = fdp_e + timedelta(minutes=15)
            # If end looks like after midnight relative to a late start, roll forward one day
            if fdp_end < s or (s.hour > 18 and fdp_end.hour < 6):
                fdp_end += timedelta(days=1)
            fdp.at[i, "FDP_min"] = (fdp_end - s).total_seconds() / 60.0

        # Duty end with post-midnight roll
        if duty_e < s or (s.hour > 18 and duty_e.hour < 6):
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
                    if fdp.loc[i-1, "fdp_end"]:
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
    dbg_cols = ["Name", "Date", "duty_start", "fdp_end", "duty_end",
                "FDP_hrs", "Duty_hrs", "Turn_hrs", "hrs7d", "hrs30d"]
    available_cols = [c for c in dbg_cols if c in fdp.columns]
    st.dataframe(fdp[available_cols], use_container_width=True)

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
