import pandas as pd
import json

# Load CSV
df = pd.read_csv("2020_samples.csv")

# Filter rows where state.timestep == 0
def parse_timestep(state_str):
    try:
        state_dict = json.loads(state_str)
        return state_dict.get("timestep", -1)
    except json.JSONDecodeError:
        return -1

# Apply function to create a new column for timestep
df["parsed_timestep"] = df["state"].apply(parse_timestep)

# Filter rows with timestep == 0
filtered_df = df[df["parsed_timestep"] == 0]

# Drop the helper column if not needed
filtered_df = filtered_df.drop(columns=["parsed_timestep"])

# Save or return
filtered_df.to_csv("2020_samples_0.csv", index=False)
