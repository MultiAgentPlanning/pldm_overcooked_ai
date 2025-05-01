from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
import pandas as pd
import ast
import json
import random

def convert_action_vector_to_symbol(action):
    direction_map = {
        (0, -1): '↑',
        (0, 1):  '↓',
        (-1, 0): '←',
        (1, 0):  '→',
        (0, 0):  'stay'
    }

    if isinstance(action, str) and action.upper() == 'INTERACT':
        return 'interact'
    elif isinstance(action, list) and len(action) == 2:
        return direction_map.get(tuple(action), 'unknown')
    return 'unknown'


# Reading start states
df = pd.read_csv("../sampled_unique_states_100.csv")

for idx, row in df.iterrows():
  print(f'Sample {idx}:')
  print('----------------------------')
  layout_name = row['layout_name']
  layout = ast.literal_eval(row['layout']) 
  state_dict = ast.literal_eval(row['state']) 

  mdp = OvercookedGridworld.from_layout_name(layout_name)
  state = OvercookedState.from_dict(state_dict)
  mdp_fn = lambda _info=None: mdp

  # Create env
  env = OvercookedEnv(mdp_generator_fn=mdp_fn, horizon=1000)

  # Set initial state manually
  env.state = state

# print('converted action:' + convert_action_vector_to_symbol([0,0]))
# print('converted action:' + convert_action_vector_to_symbol([0,1]))

# print('converted action:' + convert_action_vector_to_symbol([1,0]))

# print('converted action:' + convert_action_vector_to_symbol([-1,0]))

# print('converted action:' + convert_action_vector_to_symbol([0,-1]))

# print('converted action:' + convert_action_vector_to_symbol('interact'))


  for i in range(50):

    print(f'Step {i}:')

    #################
    legal_actions = mdp.get_actions(env.state)

    legal_actions_all = ['↑', '↓', '→', '←', 'stay', 'interact']

    # Choose random legal actions
    a0 = random.choice(legal_actions[0])
    a1 = random.choice(legal_actions[1])

    # Print for debugging
    print("Legal Agent 0:", [Action.ACTION_TO_CHAR[a] for a in legal_actions[0]])
    print("Legal Agent 1:", [Action.ACTION_TO_CHAR[a] for a in legal_actions[1]])
    print("Taking actions:", Action.ACTION_TO_CHAR[a0], Action.ACTION_TO_CHAR[a1])
    #########

    # Step
    next_state, reward, done, info = env.step((a0, a1))

    print(f"\nStep {i+1}:")
    # print("State:\n", env)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)

    if done:
      break





