import pandas as pd
import json

# Load the CSVs
start_df = pd.read_csv("2020_samples_0.csv")
full_df = pd.read_csv("data/raw/2020_hh_trials.csv")

# Normalize state JSONs to ensure consistent comparison
def clean_state(s):
    try:
        return json.dumps(json.loads(s), sort_keys=True)
    except:
        return None

start_df["clean_state"] = start_df["state"].apply(clean_state)
full_df["clean_state"] = full_df["state"].apply(clean_state)

# Collect all trajectories
trajectory_dfs = []
missing_trajectories = []

for idx, row in start_df.iterrows():
    target_state = row["clean_state"]

    # Find where this trajectory starts in full_df
    match_idxs = full_df.index[(full_df["clean_state"] == target_state)]

    if len(match_idxs) == 0:
        print(f"[Warning] No match for trajectory {idx}")
        missing_trajectories.append(idx)
        continue

    start_idx = match_idxs[0]
    traj_rows = [full_df.iloc[start_idx]]

    # Walk forward until the next trajectory starts
    for i in range(start_idx + 1, len(full_df)):
        if full_df.iloc[i]["cur_gameloop"] == 1:
            break
        traj_rows.append(full_df.iloc[i])

    traj_df = pd.DataFrame(traj_rows)
    traj_df["trajectory_id"] = idx  # Add an ID to distinguish

    trajectory_dfs.append(traj_df)

# Combine all
all_trajs_df = pd.concat(trajectory_dfs, ignore_index=True)

# Drop helper column
all_trajs_df.drop(columns=["clean_state"], inplace=True)

# Save
all_trajs_df.to_csv("expert_trajectories.csv", index=False)
print(f"✅ Saved 34 trajectories to all_34_trajectories.csv (missing: {missing_trajectories})")
