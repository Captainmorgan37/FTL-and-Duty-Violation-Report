import pandas as pd
from datetime import datetime

file_path = "/mnt/data/FTL Report ABA.xlsx"

# Load sheet
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Carry Name forward
df["Name"] = df["Name"].ffill()

# Parse date column with dayfirst
df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")

# Grab only key columns for inspection
cols = ["Name", "Date", "Date_parsed", "AC", "Duty", "Start Duty", "Blocks Off", "Ldg", "Blocks On"]
df_preview = df[cols].head(15)

df_preview
