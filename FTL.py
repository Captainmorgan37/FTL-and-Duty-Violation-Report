import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="FRMS / Duty Exceedance Checker", layout="wide")
st.title("FRMS / Duty Exceedance Checker")

st.markdown("""
Upload your **FL3XX FTL report**.  
This app highlights exceedances in the same format as your manual template:
1. 2 Consecutive <11h turns
2. 3 Consecutive 12h duty days
3. >40h in 7 days (unless <70h in 30 days)
4. Any other FRMS exceedance
""")

ftl_file = st.file_uploader("Upload FL3XX FTL CSV", type=["csv"])

# --- Helpers ---
def parse_time(s):
    s = str(s).strip()
    try:
        return datetime.strptime(s, "%H:%M").time()
    except Exception:
        return None

def to_dt(d, t):
    return None if pd.isna(d) or t is None else datetime.combine(d, t)

# --- Main ---
if ftl_file is not None:
    ftl = pd.read_csv(ftl_file, encoding="utf-8", engine="python")

    # Flexible date parsing
    ftl["Date_parsed"] = pd.to_datetime(ftl["Date"], errors="coerce").dt.date
    ftl["StartDuty_t"] = ftl["Start Duty"].apply(parse_time)
    ftl["BlocksOn_t"] = ftl["Blocks On"].apply(parse_time)

    # Aggregate per pilot/day (1 row per duty day)
    grp = ftl.groupby(["Name", "Date_parsed"]).agg(
        duty_start=("StartDuty_t", "min"),
        blocks_on=("BlocksOn_t", "max"),
    ).reset_index()

    # FDP calc (Start Duty → Blocks On + 15min)
    FDP_min = []
    for _, r in grp.iterrows():
        s = to_dt(r["Date_parsed"], r["duty_start"])
        e = to_dt(r["Date_parsed"], r["blocks_on"])
        if s and e:
            e = e + timedelta(minutes=15)
            if e < s:
                e += timedelta(days=1)
            FDP_min.append((e - s).total_seconds() // 60)
        else:
            FDP_min.append(None)
    grp["FDP_min"] = FDP_min

    # Merge back 7d / 30d columns from original FTL file
    if "7d" in ftl.columns and "30d" in ftl.columns:
        ftl_totals = ftl.groupby(["Name", "Date_parsed"]).agg(
            hrs7d=("7d", "last"),
            hrs30d=("30d", "last")
        ).reset_index()
        grp = pd.merge(grp, ftl_totals, on=["Name", "Date_parsed"], how="left")

    # Calculate next-day turn
    grp = grp.sort_values(["Name", "Date_parsed"])
    grp["next_start"] = grp.groupby("Name")["duty_start"].shift(-1)
    grp["next_date"] = grp.groupby("Name")["Date_parsed"].shift(-1)

    turn_min = []
    for _, r in grp.iterrows():
        if pd.isna(r["next_date"]) or r["blocks_on"] is None or r["next_start"] is None:
            turn_min.append(None)
            continue
        end_dt = to_dt(r["Date_parsed"], r["blocks_on"]) + timedelta(minutes=15)
        start_dt = to_dt(r["next_date"], r["next_start"])
        turn_min.append((start_dt - end_dt).total_seconds() // 60 if start_dt and end_dt else None)
    grp["turn_min"] = turn_min

    # --- Exceedance buckets ---
    ex_2consec, ex_3consec, ex_40in7, ex_other = [], [], [], []

    # 1) Two consecutive <11h turns
    grp["short_turn"] = grp["turn_min"] < 11*60
    for name, g in grp.groupby("Name"):
        g = g.sort_values("Date_parsed")
        consec = 0
        for _, r in g.iterrows():
            if r["short_turn"]:
                consec += 1
                if consec == 2:
                    ex_2consec.append(f"{name} — {r['Date_parsed']} (Turn {r['turn_min']/60:.2f}h)")
            else:
                consec = 0

    # 2) Three consecutive ≥12h FDPs
    for name, g in grp.groupby("Name"):
        g = g.sort_values("Date_parsed")
        consec = 0
        for _, r in g.iterrows():
            if r["FDP_min"] and r["FDP_min"] >= 12*60:
                consec += 1
                if consec == 3:
                    ex_3consec.append(f"{name} — {r['Date_parsed']} (FDP {r['FDP_min']/60:.2f}h)")
            else:
                consec = 0

    # 3) >40h in 7 days (unless <70h in 30 days) — use FL3XX values
    if "hrs7d" in grp.columns and "hrs30d" in grp.columns:
        for _, r in grp.iterrows():
            try:
                hrs7 = float(r["hrs7d"])
                hrs30 = float(r["hrs30d"])
            except Exception:
                continue
            if hrs7 > 40 and hrs30 >= 70:
                ex_40in7.append(f"{r['Name']} — {r['Date_parsed']} (7d {hrs7:.1f}h, 30d {hrs30:.1f}h)")

    # 4) Other exceedances (catch-all baseline: FDP >14h, turn <10h)
    for _, r in grp.iterrows():
        if r["FDP_min"] and r["FDP_min"] > 14*60:
            ex_other.append(f"{r['Name']} — {r['Date_parsed']} (FDP {r['FDP_min']/60:.2f}h)")
        if r["turn_min"] and r["turn_min"] < 10*60:
            ex_other.append(f"{r['Name']} — {r['Date_parsed']} (Turn {r['turn_min']/60:.2f}h)")

    # --- Display ---
    st.subheader("Potential Duty Exceedances")

    with st.expander("2 Consecutive <11 hour turns (Rest Before FDP)", expanded=True):
        if ex_2consec:
            for e in ex_2consec:
                st.write("- " + e)
        else:
            st.success("No exceedances found.")

    with st.expander("3 Consecutive 12-hour duty days", expanded=True):
        if ex_3consec:
            for e in ex_3consec:
                st.write("- " + e)
        else:
            st.success("No exceedances found.")

    with st.expander("Greater than 40h in 7 days (if NOT ok <70h/30d)", expanded=True):
        if ex_40in7:
            for e in ex_40in7:
                st.write("- " + e)
        else:
            st.success("No exceedances found.")

    with st.expander("Any other exceedance as listed in FRMS", expanded=True):
        if ex_other:
            for e in ex_other:
                st.write("- " + e)
        else:
            st.success("No exceedances found.")

    # --- Debug table ---
    st.subheader("Debug Data (per pilot/day)")
    debug = grp[["Name", "Date_parsed", "FDP_min", "turn_min", "hrs7d", "hrs30d"]].copy()
    debug["FDP_min"] = debug["FDP_min"].apply(lambda x: f"{x/60:.2f}h" if pd.notna(x) else None)
    debug["turn_min"] = debug["turn_min"].apply(lambda x: f"{x/60:.2f}h" if pd.notna(x) else None)
    st.dataframe(debug, use_container_width=True)
