# ----  solution/value/algos.py  ----------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import d3rlpy
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.q_functions import DiscreteMeanQFunctionForwarder
from d3rlpy.models.encoders import PixelEncoderFactory

from solution.value.model import QValueNet
from solution.value.config import SUPPORTED_ALGORITHMS, CNN_FILTERS, CNN_FEATURE_SIZE


@dataclass
class CustomQFactory(QFunctionFactory):
    obs_shape: tuple = None
    share_encoder: bool = False

    def __init__(self, obs_shape=None, share_encoder=False):
        self.obs_shape = tuple(obs_shape) if obs_shape is not None else None
        self.share_encoder = share_encoder

    def create_discrete(self, encoder, hidden_size, action_size):
        if self.obs_shape is None:
            raise ValueError("CustomQFactory requires obs_shape to be set.")
        q = QValueNet(self.obs_shape, action_size)
        fwd = DiscreteMeanQFunctionForwarder(q, action_size)
        return q, fwd

    @staticmethod
    def get_type() -> str:
        return "custom_pixel_q"

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        # These two keys will be fed back into __init__
        # The obs_shape needs to be a list to be JSON serializable
        return {
            "obs_shape": list(self.obs_shape) if self.obs_shape is not None else None,
            "share_encoder": self.share_encoder,
        }


def get_algo_config(algo_name: str, obs_shape, args):
    AlgoCfg = SUPPORTED_ALGORITHMS[algo_name]
    cfg = AlgoCfg()

    pixel_encoder = PixelEncoderFactory(
        filters=CNN_FILTERS,
        feature_size=CNN_FEATURE_SIZE,
        use_batch_norm=args.use_batch_norm,
    )

    cfg.encoder_factory = pixel_encoder
    
    # For algorithms that have separate actor and critic encoders
    if hasattr(cfg, 'actor_encoder_factory'):
        cfg.actor_encoder_factory = pixel_encoder
    if hasattr(cfg, 'critic_encoder_factory'):
        cfg.critic_encoder_factory = pixel_encoder

    cfg.q_func_factory = CustomQFactory(tuple(obs_shape))

    # Common parameters for all algorithms
    cfg.batch_size = args.batch_size
    cfg.learning_rate = args.learning_rate
    cfg.gamma = args.gamma
    
    # Algorithm-specific parameters
    if algo_name == "cql":
        cfg.alpha = args.cql_alpha
    elif algo_name == "bcq":
        cfg.action_flexibility = args.bcq_flexibility
        cfg.beta = args.bcq_beta
    elif algo_name in ["dqn", "double_dqn"]:
        cfg.target_update_interval = args.target_update_interval
    elif algo_name == "discrete_sac":
        cfg.alpha = args.sac_alpha
        if hasattr(args, "sac_temperature"):
            cfg.temperature = args.sac_temperature
    
    return cfg