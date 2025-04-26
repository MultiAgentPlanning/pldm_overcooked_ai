import argparse
import traceback
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.dataset import MDPDataset
import os


try:
    from solution.value.config import SUPPORTED_ALGORITHMS
    from solution.value.data_utils import find_max_dimensions, process_data
    from solution.value.algos import get_algo_config

    print("Successfully imported local modules (config, data_utils, algos).")
except ImportError as e:
    print(f"FATAL ERROR: Error importing local modules: {e}")
    traceback.print_exc()
    exit()


def main(args):
    print(f"Setting random seed: {args.seed}")
    d3rlpy.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading data from {args.csv_path}...")
    if not os.path.exists(args.csv_path):
        print(f"FATAL ERROR: CSV file not found at {args.csv_path}")
        return
    try:
        df = pd.read_csv(args.csv_path)
        required_cols = ["state", "joint_action", "reward", "trial_id", "cur_gameloop"]
        if not all(col in df.columns for col in required_cols) or df.empty:
            raise ValueError(f"CSV missing required columns or empty: {required_cols}")
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        return

    try:
        H_max, W_max = find_max_dimensions(df)
    except Exception as e:
        print(f"FATAL ERROR during dimension finding: {e}")
        traceback.print_exc()
        return

    try:
        obs, acts, rews, next_obs, terms = process_data(
            df, H_max, W_max, cache_dir=args.cache_dir
        )
    except Exception as e:
        print(f"FATAL ERROR during data processing: {e}")
        traceback.print_exc()
        return
    if len(obs) == 0:
        print("FATAL ERROR: No transitions processed.")
        return

    try:
        dataset = MDPDataset(
            observations=obs, actions=acts, rewards=rews, terminals=terms
        )
    except Exception as e:
        print(f"FATAL ERROR creating MDPDataset: {e}")
        traceback.print_exc()
        return

    obs_shape = obs.shape[1:]
    print(f"Determined raw obs_shape: {obs_shape}")

    if args.test_split > 0.0:
        _, test_eps = train_test_split(
            dataset.episodes, test_size=args.test_split, random_state=args.seed
        )
    else:
        test_eps = []

    algo_name = args.algorithm.lower()
    print(f"Configuring algorithm: {algo_name}")
    try:
        algo_config = get_algo_config(algo_name, obs_shape, args)
        # print("DEBUG factory params:", algo_config.q_func_factory.get_params())
    except Exception as e:
        print(f"FATAL ERROR building algo config: {e}")
        traceback.print_exc()
        return

    use_gpu = args.gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    print(f"Attempting to use device: {device.upper()}")
    algo = algo_config.create(device=device)
    algo.build_with_dataset(dataset)

    evaluators = (
        {
            "value_estimate": d3rlpy.metrics.InitialStateValueEstimationEvaluator(
                test_eps
            )
        }
        if test_eps
        else {}
    )
    run_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{algo_name}_{run_ts}")
    os.makedirs(log_dir, exist_ok=True)

    algo.fit(
        dataset,
        n_steps=args.steps,
        n_steps_per_epoch=args.steps_per_epoch,
        evaluators=evaluators or None,
        experiment_name=f"overcooked_{algo_name}",
        save_interval=args.save_interval,
        show_progress=True,
        # logdir=log_dir
    )

    final_model = os.path.join(log_dir, "model_final.d3")
    algo.save_model(final_model)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("algorithm", choices=list(SUPPORTED_ALGORITHMS.keys()))
    p.add_argument("--cache_dir", default="./data/")
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default="./d3rlpy_logs/")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--use_batch_norm", action="store_true")
    
    # CQL hyperparameters
    p.add_argument("--cql_alpha", type=float, default=5.0)
    
    # BCQ hyperparameters
    p.add_argument("--bcq_flexibility", type=float, default=0.1)
    p.add_argument("--bcq_beta", type=float, default=0.5)
    
    # DQN/DoubleDQN hyperparameters
    p.add_argument("--target_update_interval", type=int, default=8000)
    
    # SAC hyperparameters
    p.add_argument("--sac_alpha", type=float, default=0.2)
    p.add_argument("--sac_temperature", type=float, default=1.0)
    
    args = p.parse_args()
    main(args)
