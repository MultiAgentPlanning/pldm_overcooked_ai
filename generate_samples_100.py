import pandas as pd

# Load cleaned CSV
df = pd.read_csv("cleaned_2019_hh_trials.csv")

# Step 0: Filter out unwanted layouts
df = df[~df['layout_name'].isin(['random3', 'random0'])]

# Step 1: Drop duplicates based on 'state'
df_unique_states = df.drop_duplicates(subset='state')

# Step 2: Sample 100 rows (or fewer if not enough unique states remain)
n = min(100, len(df_unique_states))
sampled_df = df_unique_states.sample(n=n, random_state=42).reset_index(drop=True)

# Step 3: Save the result
sampled_df.to_csv("sampled_unique_states_100.csv", index=False)
print(f"Saved {n} unique state rows to 'sampled_unique_states_100.csv' (excluding random0 and random3)")
