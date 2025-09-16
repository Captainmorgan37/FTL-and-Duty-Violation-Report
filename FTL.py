import pandas as pd
from datetime import datetime, time

# ---------- Helpers ----------
def excel_time_to_time(val):
    """Convert Excel-style hh:mm:ss into datetime.time"""
    if pd.isna(val):
        return None
    # Case 1: Excel "Timestamp" anchored at 1900-01-01
    if isinstance(val, pd.Timestamp):
        if val.year == 1900:
            return time(val.hour, val.minute, val.second)
        return val.time()
    # Case 2: Timedelta
    if isinstance(val, pd.Timedelta):
        total_seconds = int(val.total_seconds())
        h, m = divmod(total_seconds // 60, 60)
        s = total_seconds % 60
        return time(h % 24, m, s)
    # Case 3: String "hh:mm:ss"
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
    """Convert Excel-style hh:mm:ss into float hours"""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.hour + val.minute/60 + val.second/3600
    if isinstance(val, pd.Timedelta):
        return val.total_seconds()/3600
    s = str(val).strip()
    if ":" in s:
        try:
            parts = [int(x) for x in s.split(":")]
            while len(parts) < 3:
                parts.append(0)
            hh, mm, ss = parts
            return hh + mm/60 + ss/3600
        except:
            return None
    try:
        return float(s)
    except:
        return None

# ---------- Parser ----------
def parse_ftl_excel(path, sheet="Sheet1"):
    df = pd.read_excel(path, sheet_name=sheet)

    # Forward-fill pilot name
    df["Name"] = df["Name"].ffill()

    # Parse date column (dd.mm.yyyy format)
    df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce").dt.date

    # Normalize key time-of-day fields
    time_cols = ["Start Duty", "Blocks Off", "Ldg", "Blocks On"]
    for col in time_cols:
        if col in df.columns:
            df[col + "_t"] = df[col].apply(excel_time_to_time)

    # Normalize rolling totals (7d, 30d) into float hours
    if "7d" in df.columns:
        df["hrs7d"] = df["7d"].apply(excel_time_to_hours)
    if "30d" in df.columns:
        df["hrs30d"] = df["30d"].apply(excel_time_to_hours)

    return df
