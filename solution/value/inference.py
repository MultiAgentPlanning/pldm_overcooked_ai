import argparse
import numpy as np
import pandas as pd
import torch
import d3rlpy
from d3rlpy.models.q_functions import register_q_func_factory
from solution.value.algos import CustomQFactory

from solution.value.data_utils import find_max_dimensions, process_data
from solution.value.config import TOTAL_JOINT_ACTIONS, NUM_ACTIONS_PER_AGENT

# Register the factory class
register_q_func_factory(CustomQFactory)

def load_model(model_path, device):
    """Load a saved d3rlpy model onto the specified device."""
    if device == 'cuda':
        device = 'cuda:0'
    return d3rlpy.load_learnable(model_path, device=device)

def predict_q_values(algo, obs, actions=None):
    """
    Predict Q-values for a given observation.
    - obs: array of shape (C,H,W) or (B,C,H,W)
    - actions: optional list of action indices; if None, predicts for all joint actions
    Returns: (qvals, actions)
    """
    arr = np.array(obs)
    if arr.ndim == 3:
        arr = arr[np.newaxis]
    if actions is None:
        acts = np.arange(TOTAL_JOINT_ACTIONS, dtype=np.int32)
        states = np.repeat(arr, TOTAL_JOINT_ACTIONS, axis=0)
        qvals = algo.predict_value(states, acts)
        return qvals, acts
    else:
        acts = np.array(actions, dtype=np.int32)
        qvals = algo.predict_value(arr, acts)
        return qvals, acts

def main(args):
    # Load and preprocess dataset
    df = pd.read_csv(args.csv_path)
    H_max, W_max = find_max_dimensions(df)
    obs, acts, rews, next_obs, terms = process_data(df, H_max, W_max, cache_dir=args.cache_dir)

    # Load model
    device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
    algo = load_model(args.model_path, device)

    # Sample first transition
    state = obs[0]             # (C,H,W)
    action = int(acts[0])      # scalar joint action
    reward = float(rews[0])    # scalar reward

    # Predict Q-values
    qvals, all_actions = predict_q_values(algo, state)
    pred_q = qvals[action]     # Q-value for the taken action

    # Display results
    print(f"Sample 0: action={action}, reward={reward:.4f}")
    print(f"Predicted Q for actual action: {pred_q:.4f}")
    best_idx = int(np.argmax(qvals))
    a0, a1 = best_idx // NUM_ACTIONS_PER_AGENT, best_idx % NUM_ACTIONS_PER_AGENT
    print(f"Best action: {best_idx} -> (A0={a0}, A1={a1}), Q={qvals[best_idx]:.4f}")
    print("All Q-values:", qvals)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for a trained d3rlpy model")
    parser.add_argument("model_path", type=str, help="Path to saved model (e.g. model_final.d3)")
    parser.add_argument("csv_path",   type=str, help="CSV used for training and inference")
    parser.add_argument("--cache_dir", type=str, default="./data/", help="Cache directory for processed data")
    parser.add_argument("--gpu",       action="store_true", help="Use GPU if available")
    args = parser.parse_args()
    main(args)