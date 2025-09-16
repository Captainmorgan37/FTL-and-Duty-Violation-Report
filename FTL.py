
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta

st.set_page_config(page_title="FTL Audit: Duty, Rest & 7d/30d Policy", layout="wide")
st.title("FTL Audit: Duty, Rest & 7d/30d Policy")

st.markdown(
    "Upload the relevant CSV exports and the app will run three checks:"
    "\n\n1) **Duty Streaks**: ≥2 and ≥3 consecutive **12+ hr duty** days _(FTL CSV)_"
    "\n2) **Short Rest**: ≥2 consecutive days with **rest < 11 hours** _(FTL CSV)_"
    "\n3) **7d/30d Policy + Detailed Duty Violations** _(Duty Violation CSV)_"
)

# -----------------------------
# Helpers
# -----------------------------
def parse_duration_to_hours(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    if s == "": return np.nan
    s = s.replace("hours", ":").replace("hour", ":").replace("H", ":").replace("h", ":")
    s = s.replace(" ", "").replace("::", ":").replace(".", ":")
    m = re.match(r"^(\d{1,3}):(\d{1,2})(?::(\d{1,2}))?$", s)
    if m:
        h=int(m.group(1)); mi=int(m.group(2)); se=int(m.group(3)) if m.group(3) else 0
        return h + mi/60 + se/3600
    m2 = re.match(r"^(\d+)\s*(m|min)$", s, flags=re.I)
    if m2: return int(m2.group(1))/60.0
    if re.match(r"^\d+(\.\d+)?$", s): return float(s)
    h = re.search(r"(\d+)\s*h", s, flags=re.I)
    mi = re.search(r"(\d+)\s*m", s, flags=re.I)
    if h or mi:
        hours=int(h.group(1)) if h else 0
        minutes=int(mi.group(1)) if mi else 0
        return hours + minutes/60
    try:
        td = pd.to_timedelta(s)
        return td.total_seconds()/3600.0
    except Exception:
        return np.nan

def try_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="utf-8")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, sep=";", engine="python", encoding="utf-8", on_bad_lines="skip")

# ---------- Column inference (FTL CSV) ----------
def infer_common_columns(df):
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

def infer_rest_column_ftl(df):
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

# ---------- Column inference (Duty Violation CSV) with robust Before/After ----------
def infer_policy_columns(dv_df):
    cols = [c.strip() for c in dv_df.columns]
    dv_df.columns = cols
    # Pilot
    pilot_candidates = [c for c in cols if re.search(r"(pilot|crew|user|employee|person|name)", c, re.I)]
    pilot_col = pilot_candidates[0] if pilot_candidates else None
    # 7d/30d
    c7_candidates = [c for c in cols if re.search(r"\b7\s*d(ay|ays)?\b|\b7d\b|past\s*7", c, re.I)]
    c30_candidates = [c for c in cols if re.search(r"\b30\s*d(ay|ays)?\b|\b30d\b|past\s*30", c, re.I)]
    if not c7_candidates:
        c7_candidates = [c for c in cols if re.search(r"(7d|past.?7).*(flight|block|time)", c, re.I)]
    if not c30_candidates:
        c30_candidates = [c for c in cols if re.search(r"(30d|past.?30).*(flight|block|time)", c, re.I)]
    c7 = c7_candidates[0] if c7_candidates else None
    c30 = c30_candidates[0] if c30_candidates else None

    # Rest After / Before (act/min) — robust patterns
    after_patterns_act = [
        r"\bRest\s*After\s*(FDP|Duty)\b.*\(act\)",
        r"\b(Post|Following)\b.*(FDP|Duty).*\(act\)",
        r"\bTurn\s*Time\s*After\b.*\(act\)"
    ]
    after_patterns_min = [
        r"\bRest\s*After\s*(FDP|Duty)\b.*\(min\)",
        r"\b(Post|Following)\b.*(FDP|Duty).*\(min\)",
        r"\bTurn\s*Time\s*After\b.*\(min\)"
    ]
    before_patterns_act = [
        r"\bRest\s*Before\s*(FDP|Duty)\b.*\(act\)",
        r"\b(Pre|Prior)\b.*(FDP|Duty).*\(act\)",
        r"\bTurn\s*Time\s*Before\b.*\(act\)"
    ]
    before_patterns_min = [
        r"\bRest\s*Before\s*(FDP|Duty)\b.*\(min\)",
        r"\b(Pre|Prior)\b.*(FDP|Duty).*\(min\)",
        r"\bTurn\s*Time\s*Before\b.*\(min\)"
    ]

    def find_col(patterns, exclude=None):
        for pat in patterns:
            for c in cols:
                if exclude and c == exclude:
                    continue
                if re.search(pat, c, re.I):
                    return c
        return None

    rest_after_act = find_col(after_patterns_act)
    rest_after_min = find_col(after_patterns_min)
    rest_before_act = find_col(before_patterns_act, exclude=rest_after_act)
    rest_before_min = find_col(before_patterns_min, exclude=rest_after_min)

    # Guard: ensure distinct matches
    if rest_before_act == rest_after_act:
        rest_before_act = find_col(before_patterns_act, exclude=rest_after_act)
    if rest_before_min == rest_after_min:
        rest_before_min = find_col(before_patterns_min, exclude=rest_after_min)

    # FDP act/max
    fdp_act = next((c for c in cols if re.search(r"(Flight\s*)?Duty\s*Period.*\(act\)|\bFDP\b.*\(act\)", c, re.I)), None)
    fdp_max = next((c for c in cols if re.search(r"(Flight\s*)?Duty\s*Period.*\(max\)|\bFDP\b.*\(max\)", c, re.I)), None)

    # Date (optional)
    date_candidates = [c for c in cols if re.search(r"(date|day)", c, re.I)]
    date_col = None
    for c in date_candidates:
        parsed = pd.to_datetime(dv_df[c], errors="coerce", dayfirst=True)
        if parsed.notna().mean() > 0.5:
            date_col = c
            break

    return {
        "pilot_col": pilot_col, "c7": c7, "c30": c30,
        "rest_after_act": rest_after_act, "rest_after_min": rest_after_min,
        "rest_before_act": rest_before_act, "rest_before_min": rest_before_min,
        "fdp_act": fdp_act, "fdp_max": fdp_max, "date_col": date_col,
        "all_cols": cols,
    }

# ---------- Generic utils ----------
def to_csv_download(df, filename, key=None):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download " + filename, data=csv_bytes, file_name=filename, mime="text/csv", key=key)

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
                    count = 1; start_date = r["Date"]
                last_date = r["Date"]
            else:
                if count >= min_consecutive and start_date is not None:
                    sequences.append({"Pilot": pilot, "StartDate": start_date.date().isoformat(),
                                      "EndDate": last_date.date().isoformat(), "ConsecutiveDays": count})
                count = 0; start_date = None; last_date = None
        if count >= min_consecutive and start_date is not None:
            sequences.append({"Pilot": pilot, "StartDate": start_date.date().isoformat(),
                              "EndDate": last_date.date().isoformat(), "ConsecutiveDays": count})
    return pd.DataFrame(sequences).sort_values(["Pilot", "StartDate"]) if sequences else pd.DataFrame(
        columns=["Pilot", "StartDate", "EndDate", "ConsecutiveDays"]
    )

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
    rest = rest.sort_values(["Pilot", "Date"]).groupby(["Pilot", "Date"], as_index=False)["RestHours"].min()
    rest["ShortRest"] = rest["RestHours"] < float(short_thresh)
    return rest

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

# -----------------------------
# Uploaders
# -----------------------------
st.sidebar.header("Uploads")
ftl_file = st.sidebar.file_uploader("FTL CSV (for Duty & Short Rest checks)", type=["csv"], key="ftl_csv")
dv_file = st.sidebar.file_uploader("Duty Violation CSV (for 7d/30d Policy + detailed checks)", type=["csv"], key="dv_csv")

# -----------------------------
# Tabs
# -----------------------------
tab_results, tab_policy, tab_debug = st.tabs(["Results (FTL)", "7d/30d Policy (Duty Violation)", "Debug"])

with tab_results:
    if ftl_file is None:
        st.info("Upload the **FTL CSV** in the sidebar to run Duty Streaks and Short Rest checks.")
    else:
        df = try_read_csv(ftl_file)
        pilot_col, date_col = infer_common_columns(df.copy())

        if not pilot_col or not date_col:
            st.error("Could not confidently identify common columns (Pilot, Date) in the FTL CSV.")
            st.write("Columns:", list(df.columns)[:60])
        else:
            duty_col = infer_duty_column(df.copy())
            rest_col = infer_rest_column_ftl(df.copy())

            if not duty_col and not rest_col:
                st.error("Could not identify Duty or Rest columns in the FTL CSV.")
            else:
                duty_work = None; rest_work = None
                if duty_col:
                    duty_work = build_duty_table(df.copy(), pilot_col, date_col, duty_col, min_hours=12.0)
                    seq2_duty = streaks(duty_work, "LongDuty", min_consecutive=2)
                    seq3_duty = streaks(duty_work, "LongDuty", min_consecutive=3)
                if rest_col:
                    rest_work = build_rest_table(df.copy(), pilot_col, date_col, rest_col, short_thresh=11.0)
                    seq2_rest = streaks(rest_work, "ShortRest", min_consecutive=2)

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
                st.subheader("Short Rest (< 11 hr)")
                if rest_work is not None and not seq2_rest.empty:
                    pilots_r = sorted(seq2_rest["Pilot"].unique().tolist())
                    st.error(f"⚠️ Short Rest TRIGGERED: {len(pilots_r)} pilot(s) with ≥2 consecutive days of rest < 11 hr: {', '.join(pilots_r)}")
                else:
                    st.success("✅ No pilots with ≥2 consecutive days of rest < 11 hr detected.")

                st.markdown("**≥ 2 consecutive days with rest < 11 hr**")
                st.dataframe(seq2_rest if rest_work is not None else pd.DataFrame(), use_container_width=True)
                if rest_work is not None:
                    to_csv_download(seq2_rest, "FTL_Consecutive_Short_Rest_Summary.csv", key="dl_rest2")

with tab_policy:
    if dv_file is None:
        st.info("Upload the **Duty Violation CSV** in the sidebar to run the 7d/30d policy screen and detailed checks.")
    else:
        dv = try_read_csv(dv_file)
        meta = infer_policy_columns(dv.copy())

        # Column mapping expander — hide Rest Before columns
        with st.expander("Column mapping (detected from CSV headers)"):
            mapping_view = {k: v for k, v in meta.items() 
                            if k not in ("all_cols", "rest_before_act", "rest_before_min")}
            st.write(mapping_view)

        pilot_col = meta["pilot_col"]; c7 = meta["c7"]; c30 = meta["c30"]
        rest_after_act = meta["rest_after_act"]; rest_after_min = meta["rest_after_min"]
        # Detected but hidden from mapping view:
        rest_before_act = meta["rest_before_act"]; rest_before_min = meta["rest_before_min"]
        fdp_act = meta["fdp_act"]; fdp_max = meta["fdp_max"]; date_col = meta["date_col"]

        if not (pilot_col and c7 and c30 and rest_after_act):
            st.error("Required columns missing in Duty Violation CSV (need: Pilot, 7d, 30d, Rest After FDP (act) at minimum).")
            st.write("Columns:", list(dv.columns)[:80])
        else:
            dv[pilot_col] = dv[pilot_col].ffill()
            cols = [pilot_col, c7, c30, rest_after_act] + \
                   ([rest_after_min] if rest_after_min else []) + \
                   ([rest_before_act] if rest_before_act else []) + \
                   ([rest_before_min] if rest_before_min else []) + \
                   ([fdp_act] if fdp_act else []) + \
                   ([fdp_max] if fdp_max else []) + \
                   ([date_col] if date_col else [])
            work = dv[cols].copy()

            new_cols = ["Pilot", "Hours7dRaw", "Hours30dRaw", "RestAfter_actRaw"]
            if rest_after_min: new_cols.append("RestAfter_minRaw")
            if rest_before_act: new_cols.append("RestBefore_actRaw")
            if rest_before_min: new_cols.append("RestBefore_minRaw")
            if fdp_act: new_cols.append("FDP_actRaw")
            if fdp_max: new_cols.append("FDP_maxRaw")
            if date_col: new_cols.append("Date")
            work.columns = new_cols

            # Parse
            p = parse_duration_to_hours
            work["Hours7d"] = work["Hours7dRaw"].map(p)
            work["Hours30d"] = work["Hours30dRaw"].map(p)
            work["RestAfter_act"] = work["RestAfter_actRaw"].map(p)
            if "RestAfter_minRaw" in work.columns: work["RestAfter_min"] = work["RestAfter_minRaw"].map(p)
            if "RestBefore_actRaw" in work.columns: work["RestBefore_act"] = work["RestBefore_actRaw"].map(p)
            if "RestBefore_minRaw" in work.columns: work["RestBefore_min"] = work["RestBefore_minRaw"].map(p)
            if "FDP_actRaw" in work.columns: work["FDP_act"] = work["FDP_actRaw"].map(p)
            if "FDP_maxRaw" in work.columns: work["FDP_max"] = work["FDP_maxRaw"].map(p)
            if "Date" in work.columns:
                work["Date"] = pd.to_datetime(work["Date"], errors="coerce", dayfirst=True)

            work = work.dropna(subset=["Pilot", "Hours7d", "Hours30d", "RestAfter_act"])

            # Aggregate conservative per date (min rest) and max rolling totals
            group_keys = ["Pilot"] + (["Date"] if "Date" in work.columns else [])
            aggdict = {"Hours7d": "max", "Hours30d": "max", "RestAfter_act": "min"}
            if "RestAfter_min" in work.columns: aggdict["RestAfter_min"] = "min"
            if "RestBefore_act" in work.columns: aggdict["RestBefore_act"] = "min"
            if "RestBefore_min" in work.columns: aggdict["RestBefore_min"] = "min"
            if "FDP_act" in work.columns: aggdict["FDP_act"] = "max"
            if "FDP_max" in work.columns: aggdict["FDP_max"] = "max"

            work = work.sort_values(group_keys).groupby(group_keys, as_index=False).agg(aggdict)

            # 7d/30d policy with turn-time
            work["Over40in7d"] = work["Hours7d"] > 40.0
            work["Under48_7d"] = work["Hours7d"] < 48.0
            work["Under70_30d"] = work["Hours30d"] < 70.0
            work["ShortTurn"] = work["RestAfter_act"] < 11.0

            def classify(row):
                if not row["Over40in7d"]:
                    return "OK (≤40 in 7d)"
                if row["Under48_7d"] and row["Under70_30d"]:
                    return "Allowed with 11h turn (PASS)" if not row["ShortTurn"] else "Allowed with 11h turn (FAIL — rest < 11h)"
                if row["Hours7d"] >= 48.0 or row["Hours30d"] >= 70.0:
                    return "Not Allowed (≥48 in 7d or ≥70 in 30d)"
                return "Review"

            work["PolicyStatus"] = work.apply(classify, axis=1)

            # Banners
            not_allowed = work[work["PolicyStatus"].str.contains("Not Allowed", na=False)]
            exception_fail = work[work["PolicyStatus"].str.contains("FAIL", na=False)]
            exception_pass = work[work["PolicyStatus"].str.contains("PASS", na=False)]

            if not not_allowed.empty or not exception_fail.empty:
                bad_pilots = sorted(set(not_allowed["Pilot"].tolist() + exception_fail["Pilot"].tolist()))
                st.error(f"⚠️ 7d/30d Policy VIOLATIONS: {len(bad_pilots)} pilot(s): {', '.join(bad_pilots)}")
            else:
                st.success("✅ 7d/30d policy clean (no 'Not Allowed' or 'Exception FAIL').")

            # Tables
            st.markdown("**Not Allowed (≥48 in 7d or ≥70 in 30d)**")
            st.dataframe(not_allowed, use_container_width=True)
            to_csv_download(not_allowed, "DutyViolation_not_allowed_ge48_7d_or_ge70_30d.csv", key="dl_na")

            st.markdown("**Over 40 in 7d — EXCEPTION FAIL (Rest < 11 h)**")
            st.dataframe(exception_fail, use_container_width=True)
            to_csv_download(exception_fail, "DutyViolation_over40_exception_FAIL_restlt11.csv", key="dl_fail")

            st.markdown("**Over 40 in 7d — EXCEPTION PASS (Rest ≥ 11 h)**")
            st.dataframe(exception_pass, use_container_width=True)
            to_csv_download(exception_pass, "DutyViolation_over40_exception_PASS_rest_ge11.csv", key="dl_pass")

            # ---- Additional detailed duty violations ----
            st.markdown("---")
            st.subheader("Additional Duty Violations")

            # Rest After vs Min (KEEP)
            if "RestAfter_min" in work.columns:
                v_after = work.dropna(subset=["RestAfter_act", "RestAfter_min"]).copy()
                v_after = v_after[v_after["RestAfter_act"] < v_after["RestAfter_min"]]
                if not v_after.empty:
                    st.error(f"⚠️ Rest After FDP violations: {len(v_after)} row(s) (act < min)")
                else:
                    st.success("✅ No Rest After FDP (act < min) violations found.")
                st.markdown("**Rest After FDP (act) < Rest After FDP (min)**")
                st.dataframe(v_after, use_container_width=True)
                to_csv_download(v_after, "Violation_RestAfter_act_lt_min.csv", key="dl_rest_after")
            else:
                st.info("Rest After FDP (min) column not found; skipping that check.")

            # FDP act vs max (KEEP)
            if "FDP_act" in work.columns and "FDP_max" in work.columns:
                v_fdp = work.dropna(subset=["FDP_act", "FDP_max"]).copy()
                v_fdp = v_fdp[v_fdp["FDP_act"] > v_fdp["FDP_max"]]
                if not v_fdp.empty:
                    st.error(f"⚠️ FDP Max violations: {len(v_fdp)} row(s) (act > max)")
                else:
                    st.success("✅ No FDP act > max violations found.")
                st.markdown("**Flight Duty Period (act) > Flight Duty Period (max)**")
                st.dataframe(v_fdp, use_container_width=True)
                to_csv_download(v_fdp, "Violation_FDP_act_gt_max.csv", key="dl_fdp")
            else:
                st.info("FDP (act/max) columns not found; skipping that check.")

            # Full export
            to_csv_download(work, "DutyViolation_with_Rest_and_DetailedChecks.csv", key="dl_all")

with tab_debug:
    st.caption("If you want coverage/normalized tables in this version's Debug tab, I can add them.")
