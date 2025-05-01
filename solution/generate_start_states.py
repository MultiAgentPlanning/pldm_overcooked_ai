import random
from collections import defaultdict
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

# Layouts and their tile counts (from earlier analysis)
layout_tile_stats = {
    "marshmallow_experiment":              {"O": (2, 2), "S": (2, 2), "P": (2, 2), "T": (4, 4), "D": (2, 2)},
    "you_shall_not_pass":                  {"O": (1, 1), "S": (16, 16), "P": (2, 2), "T": (1, 1), "D": (2, 2)},
    "cramped_corridor":                    {"O": (1, 1), "S": (2, 2), "P": (2, 2), "T": (1, 1), "D": (2, 2)},
    "marshmallow_experiment_coordination": {"O": (1, 1), "S": (1, 1), "P": (2, 2), "T": (4, 4), "D": (1, 1)},
    "asymmetric_advantages_tomato":        {"O": (2, 2), "S": (2, 2), "P": (1, 1), "T": (2, 2), "D": (2, 2)},
    "soup_coordination":                   {"O": (2, 2), "S": (2, 2), "P": (1, 1), "T": (2, 2), "D": (2, 2)},
    "counter_circuit":                     {"O": (1, 1), "S": (1, 1), "P": (2, 2), "T": (1, 1), "D": (1, 1)},
    "inverse_marshmallow_experiment":      {"O": (2, 2), "S": (2, 2), "P": (1, 1), "T": (2, 2), "D": (1, 1)},
}

# Orientation symbol mapping
orientation_char = {
    (0, -1): "↑",  # NORTH
    (0, 1):  "↓",  # SOUTH
    (1, 0):  "→",  # EAST
    (-1, 0): "←"   # WEST
}

def render_state(mdp, state):
    grid = [row[:] for row in mdp.terrain_mtx]  # Deep copy
    for i, player in enumerate(state.players):
        x, y = player.position
        symbol = orientation_char.get(player.orientation, "?") + str(i)
        grid[y][x] = symbol
    return "\n" + "\n".join("   ".join(f"{cell:>2}" for cell in row) for row in grid)


# Main loop
for layout_name, tile_stats in layout_tile_stats.items():
    print(f"\n========== Layout: {layout_name} ==========")
    for tile, (min_c, max_c) in tile_stats.items():
        print(f"  {tile}: min={min_c}, max={max_c}")

    # Setup environment
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    env = OvercookedEnv(mdp, horizon=100)

    # Valid empty cells for random player placement
    all_empty_cells = [(x, y) for y, row in enumerate(mdp.terrain_mtx)
                       for x, val in enumerate(row) if val == ' ']

    seen = set()
    count = 0
    trials = 0

    while count < 2 and trials < 5000:
        pos1, pos2 = random.sample(all_empty_cells, 2)
        key = tuple(sorted([pos1, pos2]))  # ensure uniqueness
        if key in seen:
            trials += 1
            continue

        seen.add(key)
        trials += 1

        # Random orientations
        dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        players = [
            PlayerState(pos1, random.choice(dirs)),
            PlayerState(pos2, random.choice(dirs))
        ]
        env.state = OvercookedState(players, objects=[])

        print(f"\n--- Start State {count + 1} ---")
        print(render_state(mdp, env.state))
        count += 1
