import pandas as pd
from datetime import datetime, time

# ---------- Helpers ----------
def excel_time_to_time(val):
    """Convert Excel-style hh:mm:ss into datetime.time"""
    if pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        if val.year == 1900:
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
    if "Name" in df.columns:
        df["Name"] = df["Name"].ffill()

    # Parse date column (dd.mm.yyyy format)
    if "Date" in df.columns:
        df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce").dt.date

    # Auto-detect time-of-day columns
    time_like_cols = []
    for col in df.columns:
        # Skip known non-time columns
        if col in ["Date", "Name"]: 
            continue
        # Check first 20 non-null values for a ":" or Timestamp
        sample = df[col].dropna().head(20)
        if not sample.empty:
            if any(isinstance(v, (pd.Timestamp, pd.Timedelta)) for v in sample):
                time_like_cols.append(col)
            elif any(":" in str(v) for v in sample.astype(str)):
                time_like_cols.append(col)

    for col in time_like_cols:
        df[col + "_t"] = df[col].apply(excel_time_to_time)

    # Auto-detect rolling/duration columns (7d, 30d, 365d, etc.)
    duration_cols = [c for c in df.columns if any(x in c.lower() for x in ["7d", "30d", "365d"])]
    for col in duration_cols:
        df["hrs_" + col] = df[col].apply(excel_time_to_hours)

    return df
