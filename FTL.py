import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="FTL/FRMS Checker", layout="wide")
st.title("AirSprint FRMS & FTL Duty Checker")

# -------------------- Helpers --------------------

def parse_time(val):
    """Convert HH:MM string to datetime.time; ignore empty cells."""
    if pd.isna(val) or val == "":
        return None
    try:
        return datetime.strptime(str(val).strip(), "%H:%M").time()
    except:
        return None

def consolidate_fdps(ftl):
    """
    Walk through rows per pilot/date, consolidate multiple legs into FDPs.
    Splits if FL3XX shows '(split)' or 'Rest'.
    """
    ftl = ftl.sort_values(["Name", "Date_parsed", "StartDuty_t"]).reset_index(drop=True)
    duty_periods = []
    current = None

    for _, row in ftl.iterrows():
        name = row["Name"]
        date = row["Date_parsed"]
        start = row["StartDuty_t"]
        end = row["BlocksOn_t"]

        # Split/rest markers in raw text
        start_raw = str(row.get("Start Duty", ""))
        end_raw = str(row.get("Blocks On", ""))
        is_split = "(split)" in start_raw
        is_rest = "Rest" in end_raw

        if current is None:
            current = {
                "Name": name,
                "Date": date,
                "duty_start": start,
                "blocks_on": end,
                "7d": row.get("7d"),
                "30d": row.get("30d"),
            }
            continue

        if name != current["Name"]:
            duty_periods.append(current)
            current = {
                "Name": name,
                "Date": date,
                "duty_start": start,
                "blocks_on": end,
                "7d": row.get("7d"),
                "30d": row.get("30d"),
            }
            continue

        # Extend FDP within same duty period
        if end:
            current["blocks_on"] = end
        if row.get("7d"): current["7d"] = row["7d"]
        if row.get("30d"): current["30d"] = row["30d"]

        # Split/rest markers → close FDP
        if is_split or is_rest:
            duty_periods.append(current)
            current = {
                "Name": name,
                "Date": date,
                "duty_start": start,
                "blocks_on": end,
                "7d": row.get("7d"),
                "30d": row.get("30d"),
            }

    if current:
        duty_periods.append(current)

    return pd.DataFrame(duty_periods)


# -------------------- File Upload --------------------

uploaded = st.file_uploader("Upload FL3XX FTL CSV", type=["csv"])
if uploaded:
    ftl = pd.read_csv(uploaded)

    # Flexible date parsing
    ftl["Date_parsed"] = pd.to_datetime(ftl["Date"], errors="coerce").dt.date

    # Parse times
    ftl["StartDuty_t"] = ftl["Start Duty"].apply(parse_time)
    ftl["BlocksOn_t"] = ftl["Blocks On"].apply(parse_time)

    # Consolidate into FDPs
    fdp_df = consolidate_fdps(ftl)

    # FDP length
    fdp_df["FDP_min"] = [
        ((datetime.combine(r["Date"], r["blocks_on"]) + timedelta(minutes=15))
         - datetime.combine(r["Date"], r["duty_start"])).total_seconds()/60
        if pd.notna(r["duty_start"]) and pd.notna(r["blocks_on"]) else None
        for _, r in fdp_df.iterrows()
    ]
    fdp_df["FDP_hrs"] = fdp_df["FDP_min"].apply(lambda x: round(x/60,2) if x else None)

    # Turn to next duty
    fdp_df = fdp_df.sort_values(["Name","Date","duty_start"]).reset_index(drop=True)
    fdp_df["Turn_min"] = None
    for i in range(1, len(fdp_df)):
        if fdp_df.loc[i,"Name"] == fdp_df.loc[i-1,"Name"]:
            prev_end = fdp_df.loc[i-1,"blocks_on"]
            cur_start = fdp_df.loc[i,"duty_start"]
            if pd.notna(prev_end) and pd.notna(cur_start):
                fdp_df.loc[i,"Turn_min"] = (
                    datetime.combine(fdp_df.loc[i,"Date"], cur_start)
                    - datetime.combine(fdp_df.loc[i-1,"Date"], prev_end)
                ).total_seconds()/60
    fdp_df["Turn_hrs"] = fdp_df["Turn_min"].apply(lambda x: round(x/60,2) if x else None)

    # -------------------- Rule Checks --------------------
    issues = []
    for _, r in fdp_df.iterrows():
        # FDP max
        if r["FDP_min"] and r["FDP_min"] > 14*60:
            issues.append((r["Name"], r["Date"], "FDP >14h", f"{r['FDP_hrs']}h"))

        # Short turn
        if r["Turn_min"] and r["Turn_min"] < 10*60:
            issues.append((r["Name"], r["Date"], "Turn <10h", f"{r['Turn_hrs']}h"))

        # 7d/30d from FL3XX counters
        try:
            hrs7 = float(str(r["7d"]).replace(",","."))
            hrs30 = float(str(r["30d"]).replace(",","."))
            if hrs7 > 40 and hrs30 >= 70:
                issues.append((r["Name"], r["Date"], "7d>40h + 30d≥70h",
                               f"7d={hrs7}, 30d={hrs30}"))
        except:
            pass

    issues_df = pd.DataFrame(issues, columns=["Name","Date","Rule","Details"])

    # -------------------- Output --------------------
    st.subheader("Consolidated FDP Table (debug)")
    st.dataframe(fdp_df)

    st.subheader("Exceedances Detected")
    if issues_df.empty:
        st.success("No exceedances found ✅")
    else:
        st.error(f"{len(issues_df)} exceedances found")
        st.dataframe(issues_df)
