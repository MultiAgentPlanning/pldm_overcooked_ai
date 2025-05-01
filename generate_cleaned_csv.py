import pandas as pd
from ast import literal_eval
import json
import numpy as np

# Step 1: Load CSV safely
df = pd.read_csv(
    "data/raw/2019_hh_trials.csv",
    on_bad_lines='skip',  # skips badly formed rows
    encoding='utf-8',
    engine='python'
)
print("CSV loaded with shape:", df.shape)

# Step 2: Safely evaluate columns that contain list-like strings
def safe_eval(x):
    try:
        return literal_eval(x)
    except (ValueError, SyntaxError):
        return np.nan

for col in ['joint_action', 'layout']:
    if col in df.columns:
        print(f"Parsing column: {col}")
        df[col] = df[col].astype(str).apply(safe_eval)

# Step 3: Safely parse 'state' column using json.loads
def safe_json_loads(x):
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return None

if 'state' in df.columns:
    print("Parsing column: state")
    df['state'] = df['state'].astype(str).apply(safe_json_loads)

# Step 4: Save cleaned output (optional)
df.to_csv("cleaned_2019_hh_trials.csv", index=False)
print("Cleaned data saved to cleaned_2019_hh_trials.csv")
