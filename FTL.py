import re
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="FTL/FRMS Checker", layout="wide")
st.title("AirSprint FRMS & FTL Duty Checker")

# ---------- Helpers ----------
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")

def parse_hhmm_from_text(val):
    if pd.isna(val): return None
    s = str(val)
    m = TIME_RE.search(s)
    if not m: return None
    hh, mm = int(m.group(1)), int(m.group(2))
    try:
        return datetime.strptime(f"{hh:02d}:{mm:02d}", "%H:%M").time()
    except: return None

def duration_to_minutes(val):
    if val is None or (isinstance(val, float) and pd.isna(val)): return None
    s = str(val).strip()
    if s == "" or s.lower() == "none": return None
    if ":" in s:
        parts = s.split(":")
        try:
            h = int(parts[0]); m = int(parts[1]) if len(parts)>1 else 0; sec = int(parts[2]) if len(parts)>2 else 0
        except: return None
        return h*60 + m + sec/60.0
    try: return float(s)*60.0
    except: return None

def minutes_to_hours(mins):
    if mins is None or (isinstance(mins,float) and pd.isna(mins)): return None
    return round(mins/60.0,2)

def to_dt(d,t): return None if pd.isna(d) or t is None else datetime.combine(d,t)

# ---------- FDP consolidation ----------
def consolidate_fdps(ftl):
    ftl = ftl.sort_values(["Name","Date_parsed","RowOrder"]).reset_index(drop=True)
    periods=[]; cur=None
    for _,row in ftl.iterrows():
        name=row["Name"]; date=row["Date_parsed"]
        sd_raw=str(row.get("Start Duty","") or ""); bo_raw=str(row.get("Blocks On","") or "")
        start_t=row["StartDuty_t"]; end_t=row["BlocksOn_t"]
        is_split="(split)" in sd_raw; is_rest="Rest" in bo_raw; is_new_start=start_t is not None
        if cur is None or name!=cur["Name"]:
            if cur is not None: periods.append(cur)
            cur={"Name":name,"Date":date,"duty_start":start_t,"blocks_on":end_t,
                 "hrs7d":row["hrs7d"],"hrs30d":row["hrs30d"],"split":False}
            continue
        if is_split or is_rest or (is_new_start and cur["duty_start"] and start_t):
            periods.append(cur)
            cur={"Name":name,"Date":date,"duty_start":start_t,"blocks_on":end_t,
                 "hrs7d":row["hrs7d"],"hrs30d":row["hrs30d"],"split":True}
            continue
        if end_t: cur["blocks_on"]=end_t
        if pd.notna(row["hrs7d"]): cur["hrs7d"]=row["hrs7d"]
        if pd.notna(row["hrs30d"]): cur["hrs30d"]=row["hrs30d"]
    if cur is not None: periods.append(cur)
    return pd.DataFrame(periods)

# ---------- File upload ----------
uploaded=st.file_uploader("Upload FL3XX FTL CSV",type=["csv"])

if uploaded:
    try: ftl=pd.read_csv(uploaded,engine="python")
    except: ftl=pd.read_csv(uploaded)

    ftl["Name"]=ftl["Name"].ffill()
    ftl["RowOrder"]=range(len(ftl))
    ftl["Date_parsed"]=pd.to_datetime(ftl["Date"],errors="coerce").dt.date
    ftl["StartDuty_t"]=ftl["Start Duty"].apply(parse_hhmm_from_text)
    ftl["BlocksOn_t"]=ftl["Blocks On"].apply(parse_hhmm_from_text)
    ftl["hrs7d"]=ftl["7d"].apply(duration_to_minutes).apply(minutes_to_hours) if "7d" in ftl.columns else pd.NA
    ftl["hrs30d"]=ftl["30d"].apply(duration_to_minutes).apply(minutes_to_hours) if "30d" in ftl.columns else pd.NA

    fdp=consolidate_fdps(ftl)

    # FDP length
    fdp["FDP_min"]=None
    for i,r in fdp.iterrows():
        s=to_dt(r["Date"],r["duty_start"]); e=to_dt(r["Date"],r["blocks_on"])
        if s and e:
            e+=timedelta(minutes=15)
            if e<s: e+=timedelta(days=1)
            fdp.at[i,"FDP_min"]=(e-s).total_seconds()/60.0
    fdp["FDP_hrs"]=fdp["FDP_min"].apply(minutes_to_hours)

    # Turns
    fdp=fdp.sort_values(["Name","Date","duty_start","blocks_on"]).reset_index(drop=True)
    fdp["Turn_min"]=None
    for i in range(1,len(fdp)):
        if fdp.loc[i,"Name"]==fdp.loc[i-1,"Name"]:
            if not fdp.loc[i,"split"]:
                prev_end=to_dt(fdp.loc[i-1,"Date"],fdp.loc[i-1,"blocks_on"])
                cur_start=to_dt(fdp.loc[i,"Date"],fdp.loc[i,"duty_start"])
                if prev_end and cur_start:
                    prev_end+=timedelta(minutes=15)
                    if cur_start<prev_end: cur_start+=timedelta(days=1)
                    fdp.at[i,"Turn_min"]=(cur_start-prev_end).total_seconds()/60.0
    fdp["Turn_hrs"]=fdp["Turn_min"].apply(minutes_to_hours)

    # ---------- Rule checks ----------
    issues=[]
    def add_issue(name,date,rule,details): issues.append({"Name":name,"Date":date,"Rule":rule,"Details":details})

    for i,r in fdp.iterrows():
        fdp_min=r["FDP_min"] if pd.notna(r["FDP_min"]) else None
        turn_min=r["Turn_min"] if pd.notna(r["Turn_min"]) else None
        hrs7=r["hrs7d"] if pd.notna(r["hrs7d"]) else None
        hrs30=r["hrs30d"] if pd.notna(r["hrs30d"]) else None

        if fdp_min is None: continue

        # FDP exceedances
        if fdp_min>15*60:
            add_issue(r["Name"],r["Date"],"FDP >15h",f"{minutes_to_hours(fdp_min)}h")
        elif fdp_min>14*60:
            # Check eligibility for 15h
            eligible=False
            if hrs30 is not None and hrs30<70: eligible=True
            # Check ≥24h rest before this FDP
            if i>0 and fdp.loc[i,"Name"]==fdp.loc[i-1,"Name"]:
                prev_turn=fdp.loc[i,"Turn_min"]
                if prev_turn and prev_turn>=24*60: eligible=True
            if not eligible:
                add_issue(r["Name"],r["Date"],"FDP >14h without eligibility",f"{minutes_to_hours(fdp_min)}h, 30d={hrs30}h")
            else:
                # Must check next rest extension
                if i+1<len(fdp) and fdp.loc[i+1,"Name"]==r["Name"]:
                    next_turn=fdp.loc[i+1,"Turn_min"]
                    if next_turn and next_turn<12*60:
                        add_issue(r["Name"],r["Date"],"Post-extension rest too short",f"Rest {minutes_to_hours(next_turn)}h")

        # Min turn <10h (normal rule)
        if turn_min and turn_min<10*60:
            add_issue(r["Name"],r["Date"],"Turn <10h",f"Turn {minutes_to_hours(turn_min)}h")

        # 7d/30d cumulative
        if hrs7 and hrs30 and hrs7>40 and hrs30>=70:
            add_issue(r["Name"],r["Date"],"7d>40h + 30d≥70h",f"7d={hrs7}h, 30d={hrs30}h")

    issues_df=pd.DataFrame(issues)

    # ---------- UI ----------
    st.subheader("Consolidated FDPs (debug)")
    dbg_cols=["Name","Date","duty_start","blocks_on","FDP_hrs","Turn_hrs","hrs7d","hrs30d","split"]
    st.dataframe(fdp[dbg_cols],use_container_width=True)

    st.subheader("Exceedances Detected")
    if issues_df.empty:
        st.success("No exceedances found ✅")
    else:
        st.error(f"{len(issues_df)} exceedance(s) found")
        st.dataframe(issues_df,use_container_width=True)
        st.download_button("Download exceedances (CSV)",
                           issues_df.to_csv(index=False),
                           file_name="exceedances.csv",
                           mime="text/csv")
