import pandas as pd
import json

# Load CSVs
start_df = pd.read_csv("2020_samples_0.csv")
full_df = pd.read_csv("data/raw/2020_hh_trials.csv")

# Preprocess: strip whitespace for consistency and parse 'state' JSONs
def clean_state(s):
    try:
        return json.dumps(json.loads(s), sort_keys=True)  # normalized JSON string
    except:
        return None

start_df["clean_state"] = start_df["state"].apply(clean_state)
full_df["clean_state"] = full_df["state"].apply(clean_state)

# Create a list of all full_df trajectories
results = []

# Iterate over each starting state
for idx, start_row in start_df.iterrows():
    start_state = start_row["clean_state"]

    # Find first match with cur_gameloop == 1
    match_idx = full_df.index[(full_df["clean_state"] == start_state) & (full_df["cur_gameloop"] == 1)]

    if len(match_idx) == 0:
        print(f"No match found for row {idx}")
        results.append(None)
        continue

    start_index = match_idx[0]
    final_index = start_index

    # Traverse forward until next cur_gameloop == 1 or end of dataframe
    for i in range(start_index + 1, len(full_df)):
        if full_df.iloc[i]["cur_gameloop"] == 1:
            break
        final_index = i

    final_score = full_df.iloc[final_index]["score"]
    results.append(final_score)

# Add results to start_df
start_df["final_score"] = results

# Drop helper column
start_df.drop(columns=["clean_state"], inplace=True)

# Save to CSV
start_df.to_csv("expert_scores.csv", index=False)
