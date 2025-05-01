import os
import pandas as pd
import ast
import json
import random
import argparse
import d3rlpy
import numpy as np
import torch

# Add to imports at the top
import csv

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

from solution.pldm.trainer import PLDMTrainer
from solution.pldm.config import load_config
from solution.pldm.planner_mppi import Planner
from solution.value.inference import OvercookedValueEstimator
from solution.pldm.data_processor import get_overcooked_dataloaders
from solution.pldm.state_encoder import GridStateEncoder, VectorStateEncoder
from solution.pldm.utils import ACTION_INDICES_REVERSE


def convert_action_vector_to_symbol(action):
    # direction_map = {
    #     (0, -1): "↑",
    #     (0, 1): "↓",
    #     (-1, 0): "←",
    #     (1, 0): "→",
    #     (0, 0): "stay",
    # }

    direction_map = {
        (0, -1): (0, -1),
        (0, 1): (0, 1),
        (-1, 0): (-1, 0),
        (1, 0): (1, 0),
        (0, 0): (0, 0),
    }

    if isinstance(action, str) and action.upper() == "INTERACT":
        return "interact"
    elif isinstance(action, list) and len(action) == 2:
        return direction_map.get(tuple(action), "unknown")
    return "unknown"


def main(args):

    # In the main function, right after creating the planner
    # Set up CSV logging
    csv_filename = os.path.join(args.output_dir, "simulation_log_2020.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_headers = [
        "sample_idx",
        "layout_name",
        "step",
        "state",
        "reward",
        "done",
        "info",
        "a0",
        "a1",
    ]

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_filename)
    if not file_exists:
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()

    # Reading start states
    df = pd.read_csv("2020_samples_unique_100.csv")
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
        grid_height=pl_config["model"].get(
            "grid_height"
        ),  # Use .get for optional params
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
        config=pl_config,  # Pass the entire config to the trainer
    )
    trainer._initialize_models()

    state_encoder_net = trainer.cnn_encoder
    s0, _, _, _ = next(iter(trainer.train_loader))  # first batch
    state_encoder_net(torch.ones_like(s0[:1]).to(device))  # 1-step forward to init

    ck_cnn = os.path.join(args.pldm_dir, "cnn_encoder.pt")
    state_encoder_net.load_state_dict(
        torch.load(ck_cnn, map_location=device)["model_state_dict"]
    )
    state_encoder_net.eval()

    dynamics_model = trainer.dynamics_model
    ck_cnn = os.path.join(args.pldm_dir, "dynamics_model.pt")
    dynamics_model.load_state_dict(
        torch.load(ck_cnn, map_location=device)["model_state_dict"]
    )
    dynamics_model.eval()

    reward_model = OvercookedValueEstimator(
        args.reward_model_path,
        args.csv_path,
        args.pldm_dir,
        6,
        device=device,
        seed=args.seed,
        trainer=trainer,
    )

    state_encoder = GridStateEncoder()

    planner = Planner(
        dynamics_model=dynamics_model,
        reward_model=reward_model,
        state_encoder=state_encoder,  # Should be GridStateEncoder or VectorStateEncoder instance
        state_encoder_net=state_encoder_net,
        planning_horizon=args.planning_horizon,
        lambda_temp=args.lambda_temp,
        device=device,
        seed=args.seed,
    )
    
    num_dones = 0

    for idx, row in df.iterrows():
        print(f"Sample {idx}:")
        print("----------------------------")
        layout_name = row["layout_name"]
        layout = ast.literal_eval(row["layout"])
        state_dict = ast.literal_eval(row["state"])

        mdp = OvercookedGridworld.from_layout_name(layout_name)
        state = OvercookedState.from_dict(state_dict)
        mdp_fn = lambda _info=None: mdp

        # Create env
        env = OvercookedEnv(mdp_generator_fn=mdp_fn, horizon=args.max_steps)

        # Set initial state manually
        env.state = state

        # print('converted action:' + convert_action_vector_to_symbol([0,0]))
        # print('converted action:' + convert_action_vector_to_symbol([0,1]))

        # print('converted action:' + convert_action_vector_to_symbol([1,0]))

        # print('converted action:' + convert_action_vector_to_symbol([-1,0]))

        # print('converted action:' + convert_action_vector_to_symbol([0,-1]))

        # print('converted action:' + convert_action_vector_to_symbol('interact'))

        for i in range(args.max_steps):

            print(f"Step {i}:")

            #################
            legal_actions = mdp.get_actions(env.state)
            print("legal actions:", legal_actions)

            # legal_actions_all = ['↑', '↓', '→', '←', 'stay', 'interact']

            # # Choose random legal actions
            # a0 = random.choice(legal_actions[0])
            # a1 = random.choice(legal_actions[1])

            # # Print for debugging
            # print("Legal Agent 0:", [Action.ACTION_TO_CHAR[a] for a in legal_actions[0]])
            # print("Legal Agent 1:", [Action.ACTION_TO_CHAR[a] for a in legal_actions[1]])
            # print("Taking actions:", Action.ACTION_TO_CHAR[a0], Action.ACTION_TO_CHAR[a1])
            
            # print("State:\n", str(env.state))
            # print("State Dict:\n", str(env.state.to_dict()))
            (a0, a1), _ = planner.plan(env.state.to_dict())
            print("Actions:", a0, a1)
            # a0 = ACTION_INDICES_REVERSE[a0]
            # a1 = ACTION_INDICES_REVERSE[a1]
            a0 = convert_action_vector_to_symbol(ACTION_INDICES_REVERSE[a0])
            a1 = convert_action_vector_to_symbol(ACTION_INDICES_REVERSE[a1])

            #########

            # Step
            state, reward, done, info = env.step((a0, a1))

            print(f"\nStep {i+1}:")
            # print("State:\n", str(env.state.to_dict()))
            print("Reward:", reward)
            print("Done:", done)
            print("Num Dones:", num_dones)
            print("Info:", info)

            with open(csv_filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                writer.writerow(
                    {
                        "sample_idx": idx,
                        "layout_name": layout_name,
                        "step": i + 1,
                        "state": str(env.state.to_dict()),
                        "reward": reward,
                        "done": done,
                        "info": str(info),
                        "a0": a0,
                        "a1": a1,
                    }
                )

            if done:
                if i != args.max_steps - 1:
                  num_dones += 1
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("output_dir")
    p.add_argument("--planning_horizon", type=float, default=10)
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--lambda_temp", type=float, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", action="store_true")
    p.add_argument(
        "--pldm_dir",
        type=str,
        required=True,
        help="Folder containing training_config.yaml and state_encoder_net.pt",
    )
    p.add_argument(
        "--reward_model_path",
        type=str,
        required=True,
        help="Folder containing training_config.yaml and state_encoder_net.pt",
    )

    args = p.parse_args()
    main(args)
