import ast
import os
import json 
import pandas as pd
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, Recipe, OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# Always configure Recipe first
Recipe.configure({})
df = pd.read_csv("simulation_log_2020_run_18_overcooked_discrete_sac_20250430193519_PH10_NS100_MS400_LT1.0.csv")

for idx, row in df.iterrows():
    sample_id = row['sample_idx']
    layout = row["layout_name"]
    state_dict = ast.literal_eval(row["state"])

    if sample_id == 12: # to compute for a specific sample
    # Convert to OvercookedState
        state = OvercookedState.from_dict(state_dict)
        mdp = OvercookedGridworld.from_layout_name(layout)
        grid = mdp.terrain_mtx
        # Render

        dir_name = f"visual_output_{sample_id}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        StateVisualizer().display_rendered_state(
            state=state,
            grid=grid,
            img_path=f"visual_output_{sample_id}/state_render{state_dict['timestep']}.png"
        )
