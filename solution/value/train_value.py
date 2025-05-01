import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import d3rlpy
from d3rlpy.dataset import MDPDataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from solution.value.config import SUPPORTED_ALGORITHMS
from solution.value.algos import get_algo_config
from solution.pldm.config import load_config
from solution.pldm.trainer import PLDMTrainer
from solution.pldm.state_encoder import StateEncoderNetwork
from solution.pldm.data_processor import OvercookedDataset, custom_collate_fn, get_overcooked_dataloaders


def main(args):
    # ---- seeding ----
    d3rlpy.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

    # ---- load config ----
    pl_config = load_config(os.path.join(args.pldm_dir, "training_config.yaml"))
    mc = pl_config["model"]

    trainer = PLDMTrainer(
        data_path=args.csv_path,
        output_dir=args.output_dir,
        model_type=pl_config["model"]["type"],
        batch_size=pl_config["training"]["batch_size"],
        lr=pl_config["training"]["learning_rate"],
        state_embed_dim=pl_config["model"]["state_embed_dim"],
        action_embed_dim=pl_config["model"]["action_embed_dim"],
        num_actions=pl_config["model"]["num_actions"],
        dynamics_hidden_dim=pl_config["model"]["dynamics_hidden_dim"],
        reward_hidden_dim=pl_config["model"]["reward_hidden_dim"],
        grid_height=pl_config["model"].get("grid_height"), # Use .get for optional params
        grid_width=pl_config["model"].get("grid_width"),
        device=device,
        max_samples=pl_config["data"].get("max_samples"),
        num_workers=pl_config["training"]["num_workers"],
        val_ratio=pl_config["data"].get("val_ratio", 0.1),
        test_ratio=pl_config["data"].get("test_ratio", 0.1),
        seed=args.seed,  # Pass the seed for reproducible data splitting
        # Pass wandb_run object to trainer if needed, or check wandb.run directly
        wandb_run=None,
        disable_artifacts=True,  # Disable WandB artifacts to avoid storage errors
        config=pl_config  # Pass the entire config to the trainer
    )
    trainer._initialize_models()
    
    cnn_encoder = trainer.cnn_encoder
    print(f"Using CNN encoder:")
    print(type(cnn_encoder))
    print(cnn_encoder)

    s0, _, _, _ = next(iter(trainer.train_loader))          # first batch
    cnn_encoder(torch.ones_like(s0[:1]).to(device))         # 1-step forward to init

    ck_cnn = os.path.join(args.pldm_dir, "cnn_encoder.pt")
    cnn_encoder.load_state_dict(torch.load(ck_cnn, map_location=device)["model_state_dict"])
    cnn_encoder.eval()

    # replace the trainer's loader with one that returns terminals
    loader, _, _ = get_overcooked_dataloaders(
        args.csv_path,
        state_encoder_type=pl_config["model"]["type"],
        batch_size=pl_config["training"]["batch_size"],
        val_ratio=pl_config["data"].get("val_ratio", 0.1),
        test_ratio=pl_config["data"].get("test_ratio", 0.1),
        max_samples=pl_config["data"].get("max_samples"),
        return_terminal=True,
        num_workers=pl_config["training"]["num_workers"],
        seed=args.seed
    )

    # ---- collect transitions ----
    obs_list, next_obs_list, act_list, rew_list, term_list = [], [], [], [], []
    with torch.no_grad():
        for state_t, act_t, next_state_t, rew_t, term_t in tqdm(loader):
            # latent embeddings
            z_t = cnn_encoder(state_t.to(device)).cpu().numpy().astype(np.float32)
            z_next = cnn_encoder(next_state_t.to(device)).cpu().numpy().astype(np.float32)
            obs_list.extend(z_t)
            next_obs_list.extend(z_next)

            # flatten joint actions
            joint = act_t[:, 0] * args.num_actions + act_t[:, 1]
            act_list.extend(joint.cpu().numpy().astype(np.int32).tolist())

            # rewards
            rew_list.extend(rew_t.cpu().numpy().astype(np.float32).tolist())

            # terminals
            term_list.extend(term_t.cpu().numpy().tolist())

    # ---- to numpy arrays ----
    obs_arr = np.array(obs_list, dtype=np.float32)
    # next_obs_arr = np.array(next_obs_list, dtype=np.float32)
    acts_arr = np.array(act_list, dtype=np.int32)
    rews_arr = np.array(rew_list, dtype=np.float32)
    terms_arr = np.array(term_list, dtype=np.float32)

    print("Dataset shapes:")
    print(f"  obs: {obs_arr.shape}")
    print(f"  actions: {acts_arr.shape}")
    print(f"  rewards: {rews_arr.shape}")
    print(f"  terminals: {terms_arr.shape}")

    # ---- create MDPDataset ----
    mdp_dataset = MDPDataset(
        observations=obs_arr,
        actions=acts_arr,
        rewards=rews_arr,
        terminals=terms_arr
    )
    obs_shape = obs_arr.shape[1:]

    # ---- split ----
    if args.test_split > 0.0:
        _, test_eps = train_test_split(
            mdp_dataset.episodes,
            test_size=args.test_split,
            random_state=args.seed
        )
    else:
        test_eps = []

    # ---- configure algorithm ----
    algo_name = args.algorithm.lower()
    algo_cfg = get_algo_config(algo_name, obs_shape, args)
    algo = algo_cfg.create(device=device)
    algo.build_with_dataset(mdp_dataset)

    evaluators = {"value_estimate": d3rlpy.metrics.InitialStateValueEstimationEvaluator(test_eps)} if test_eps else {}

    # ---- training ----
    run_ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, f"{algo_name}_{run_ts}")
    os.makedirs(log_dir, exist_ok=True)

    algo.fit(
        mdp_dataset,
        n_steps=args.steps,
        n_steps_per_epoch=args.steps_per_epoch,
        evaluators=evaluators or None,
        experiment_name=f"overcooked_{algo_name}",
        save_interval=args.save_interval,
        show_progress=True
    )

    algo.save_model(os.path.join(log_dir, "model_final.d3"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("output_dir")
    p.add_argument("algorithm", choices=list(SUPPORTED_ALGORITHMS.keys()))
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default="./d3rlpy_logs/")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--learning_rate", "--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--save_interval", type=int, default=10)
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--pldm_dir", type=str, required=True,
        help="Folder containing training_config.yaml and cnn_encoder.pt"
    )
    p.add_argument("--num_actions", type=int, default=6,
                   help="Number of actions per agent for flattening joint action")
    p.add_argument("--use_batch_norm", action="store_true",
                   help="Enable batch normalization in the pixel encoder")

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
