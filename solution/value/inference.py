import argparse, os, re, numpy as np, torch, d3rlpy
from pathlib import Path
from tqdm import tqdm

from d3rlpy.models.q_functions import register_q_func_factory
from solution.value.algos import CustomQFactory  # <-- ensures deserialization works
from solution.pldm.config import load_config
from solution.pldm.trainer import PLDMTrainer
from solution.pldm.data_processor import get_overcooked_dataloaders
from solution.value.config import NUM_ACTIONS_PER_AGENT, TOTAL_JOINT_ACTIONS

# register factory once so ``load_learnable`` can resolve "custom_pixel_q"
register_q_func_factory(CustomQFactory)

# --------------------------------------------------------------------------- #
#                              helper routines                                #
# --------------------------------------------------------------------------- #

def _find_model_file(path: str | Path, step: int | None) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    patt = re.compile(r"model_(\d+)\.d3$")
    ckpts = [(int(m.group(1)), f) for f in p.glob("model_*.d3") if (m := patt.match(f.name))]
    if not ckpts:
        raise FileNotFoundError(f"No *.d3 checkpoint files found in {p}")
    ckpts.sort(key=lambda t: t[0])
    if step is None:
        return ckpts[-1][1]
    for s, f in ckpts:
        if s == step:
            return f
    raise FileNotFoundError(f"model_{step}.d3 not found in {p}")


def _load_encoder(csv_path, pldm_dir, device, seed):
    cfg = load_config(Path(pldm_dir, "training_config.yaml"))
    tr = PLDMTrainer(csv_path, "/tmp", cfg["model"]["type"], 1, 1e-4,
                     cfg["model"]["state_embed_dim"], cfg["model"]["action_embed_dim"],
                     cfg["model"]["num_actions"], cfg["model"]["dynamics_hidden_dim"],
                     cfg["model"]["reward_hidden_dim"], device=device,
                     disable_artifacts=True, config=cfg, seed=seed)
    tr._initialize_models()
    enc = tr.cnn_encoder.to(device)
    enc(next(iter(tr.train_loader))[0][:1].to(device))
    enc.load_state_dict(torch.load(Path(pldm_dir, "cnn_encoder.pt"), map_location=device)["model_state_dict"])
    enc.eval()
    return enc


class OvercookedValueEstimator:
    """Q / V estimator for Overcooked grid states."""
    def __init__(self, model_path_or_dir, csv_path, pldm_dir,
                 num_actions=6, step=None, device="cpu", seed=0):
        self.device = device
        self.N = num_actions
        self.Ntot = num_actions * num_actions
        self.encoder = _load_encoder(csv_path, pldm_dir, device, seed)
        ckpt = _find_model_file(model_path_or_dir, step)
        print(f"[info] using checkpoint {ckpt}")
        self.policy = d3rlpy.load_learnable(ckpt, device=("cuda:0" if device == "cuda" else "cpu"))

    def _latent(self, grid):
        t = torch.tensor(grid[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.encoder(t).cpu().numpy().astype(np.float32)

    def q_all(self, grid):
        z = self._latent(grid)
        acts = np.arange(self.Ntot, dtype=np.int32)
        return self.policy.predict_value(np.repeat(z, self.Ntot, 0), acts)

    def q(self, grid, idx):
        return float(self.q_all(grid)[idx])

    def v(self, grid):
        return float(self.q_all(grid).max())

    def probs(self, grid, temp=1.0):
        q = self.q_all(grid) / temp
        e = np.exp(q - q.max())
        return e / e.sum()


def demo(args):
    dev = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"
    est = OvercookedValueEstimator(args.model_path, args.csv_path, args.pldm_dir,
                                   args.num_actions, step=args.step, device=dev, seed=args.seed)
    loader, _, _ = get_overcooked_dataloaders(args.csv_path, state_encoder_type="grid",
                                              batch_size=1, val_ratio=0, test_ratio=0,
                                              return_terminal=False, num_workers=0)
    grid, act, _, _ = next(iter(loader))
    grid = grid[0].numpy(); logged = int(act[0,0]*args.num_actions + act[0,1])
    qvals = est.q_all(grid); probs = est.probs(grid)
    print(f"logged a={logged} | Q={qvals[logged]:.4f} | V={qvals.max():.4f}\n")
    print("idx A0 A1   Q       p")
    for idx in range(est.Ntot):
        a0,a1 = divmod(idx, est.N)
        print(f"{idx:2d} {a0:2d} {a1:2d} {qvals[idx]:7.3f} {probs[idx]:5.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("model_path")
    ap.add_argument("--pldm_dir", required=True)
    ap.add_argument("--step", type=int)
    ap.add_argument("--num_actions", type=int, default=6)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    demo(ap.parse_args())
