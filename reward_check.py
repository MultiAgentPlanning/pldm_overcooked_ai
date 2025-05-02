import pandas as pd

# Load your CSV file
df = pd.read_csv("output_dir/simulation_log_2020_run_18_overcooked_discrete_sac_20250430193519_PH1_NS1_MS400_LT1.0.csv")

# Group by sample_idx and sum the reward
reward_sums = df.groupby("sample_idx")["reward"].sum()

# Print the results
print(reward_sums)
