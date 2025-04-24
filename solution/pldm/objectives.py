import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import functional as F
from typing import NamedTuple, Optional, Dict, Any, Union, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class MSELossInfo(NamedTuple):
    """Information returned by MSE loss computation."""
    total_loss: torch.Tensor
    loss_name: str = "mse"
    name_prefix: str = ""

    def build_log_dict(self) -> Dict[str, float]:
        """Build a dictionary of values to log."""
        return {
            f"{self.name_prefix}/{self.loss_name}_loss": self.total_loss.item(),
        }


class MSELoss(nn.Module):
    """Standard Mean Squared Error loss for PLDM models."""
    
    def __init__(self, name_prefix: str = "", model_type: str = "dynamics"):
        """
        Initialize MSE loss.
        
        Args:
            name_prefix: Prefix for logging names
            model_type: Type of model this loss is for ('dynamics' or 'reward')
        """
        super().__init__()
        self.name_prefix = name_prefix
        self.model_type = model_type
        self.criterion = nn.MSELoss()
        
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> MSELossInfo:
        """
        Compute MSE loss between output and target.
        
        Args:
            output: Model prediction
            target: Ground truth
            
        Returns:
            MSELossInfo containing the loss
        """
        # Ensure shapes match
        if output.shape != target.shape:
            # For dynamics model, sometimes we need to interpolate
            if self.model_type == 'dynamics':
                output = F.interpolate(output, size=target.shape[2:], mode='nearest')
            else:
                # For reward model, ensure dimensions match
                if output.dim() == 1: 
                    output = output.unsqueeze(1)
                if target.dim() == 1:
                    target = target.unsqueeze(1)
        
        loss = self.criterion(output, target)
        
        return MSELossInfo(
            total_loss=loss,
            loss_name=f"mse_{self.model_type}",
            name_prefix=self.name_prefix
        )


@dataclass
class VICRegConfig:
    """Configuration for VICReg loss."""
    # Projection settings
    projector_layers: List[int] = None  # List of hidden layer dimensions for projector
    output_dim: int = 256  # Output dimension of projector
    projector_type: str = "mlp"  # Type of projector: "mlp" or "identity"
    
    # Loss coefficients
    sim_coeff: float = 25.0  # Coefficient for similarity loss
    std_coeff: float = 25.0  # Coefficient for standard deviation loss
    cov_coeff: float = 1.0   # Coefficient for covariance loss
    
    # Temporal coefficients (if using temporal data)
    sim_coeff_t: float = 0.0  # Coefficient for temporal similarity loss
    std_coeff_t: float = 0.0  # Coefficient for temporal standard deviation loss
    cov_coeff_t: float = 0.0  # Coefficient for temporal covariance loss
    
    # Other settings
    std_margin: float = 1.0  # Margin for standard deviation loss
    std_margin_t: float = 1.0  # Margin for temporal standard deviation loss
    adjust_cov: bool = True  # Whether to adjust covariance loss by dividing by (feature_dim - 1)


class VICRegLossInfo(NamedTuple):
    """Information returned by VICReg loss computation."""
    total_loss: torch.Tensor
    sim_loss: torch.Tensor
    std_loss: torch.Tensor
    cov_loss: torch.Tensor
    std_loss_t: torch.Tensor
    cov_loss_t: torch.Tensor
    sim_loss_t: torch.Tensor
    loss_name: str = "vicreg"
    name_prefix: str = ""

    def build_log_dict(self) -> Dict[str, float]:
        """Build a dictionary of values to log."""
        return {
            f"{self.name_prefix}/{self.loss_name}_total_loss": self.total_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_sim_loss": self.sim_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_std_loss": self.std_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_cov_loss": self.cov_loss.item(),
            f"{self.name_prefix}/{self.loss_name}_std_loss_t": self.std_loss_t.item(),
            f"{self.name_prefix}/{self.loss_name}_cov_loss_t": self.cov_loss_t.item(),
            f"{self.name_prefix}/{self.loss_name}_sim_loss_t": self.sim_loss_t.item(),
        }


class Projector(nn.Module):
    """Projector network for VICReg."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, output_dim: int = 256, projector_type: str = "mlp"):
        """
        Initialize projector network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            projector_type: Type of projector ("mlp" or "identity")
        """
        super().__init__()
        
        self.projector_type = projector_type.lower()
        
        if projector_type.lower() == "identity":
            # Identity projector - no parameters, just passes through the input
            self.projector = nn.Identity()
            logger.info("Using identity projector for VICReg")
        else:
            # Default MLP projector
            if hidden_dims is None:
                hidden_dims = [2048, 2048, 2048]
            
            layers = []
            in_dim = input_dim
            
            # Create hidden layers
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                in_dim = hidden_dim
            
            # Add final layer
            layers.append(nn.Linear(in_dim, output_dim))
            
            self.projector = nn.Sequential(*layers)
            logger.info(f"Using MLP projector with dimensions: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Flatten if needed (unless using identity projector)
        if self.projector_type != "identity" and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        return self.projector(x)


class VICRegLoss(nn.Module):
    """
    VICReg loss implementation for PLDM models.
    Based on the paper "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
    """
    
    def __init__(
        self,
        config: VICRegConfig,
        input_dim: int,
        name_prefix: str = "",
        model_type: str = "dynamics"
    ):
        """
        Initialize VICReg loss.
        
        Args:
            config: Configuration for VICReg loss
            input_dim: Input dimension for projector
            name_prefix: Prefix for logging names
            model_type: Type of model this loss is for ('dynamics' or 'reward')
        """
        super().__init__()
        self.config = config
        self.name_prefix = name_prefix
        self.model_type = model_type
        
        # Flatten input_dim if it's not an integer
        if isinstance(input_dim, (tuple, list)):
            input_dim = torch.prod(torch.tensor(input_dim)).item()
        
        # Initialize projector
        self.projector = Projector(
            input_dim=input_dim,
            hidden_dims=config.projector_layers,
            output_dim=config.output_dim,
            projector_type=config.projector_type
        )
        
        # Base MSE loss for similarity
        self.mse_loss = nn.MSELoss()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> VICRegLossInfo:
        """
        Compute VICReg loss between output and target.
        
        Args:
            output: Model prediction
            target: Ground truth
            
        Returns:
            VICRegLossInfo containing all loss components
        """
        # Basic similarity loss (MSE) between output and target
        sim_loss = self.mse_loss(output, target)
        
        # Project the representations
        # When identity projector is used, we need to check shapes
        if isinstance(self.projector.projector, nn.Identity) and output.dim() > 2:
            # For ConvNet-based architectures, we need to flatten the output
            batch_size = output.size(0)
            output_flat = output.reshape(batch_size, -1)
            target_flat = target.reshape(batch_size, -1)
            
            output_proj = self.projector(output_flat)
            target_proj = self.projector(target_flat)
        else:
            # Use normal projection
            output_proj = self.projector(output)
            target_proj = self.projector(target)
        
        # Compute std loss for regular representations
        std_loss = self._std_loss(output_proj, across_time=False)
        
        # Compute cov loss for regular representations
        cov_loss = self._cov_loss(output_proj, across_time=False)
        
        # Temporal loss components
        
        # For temporal similarity loss (sim_coeff_t)
        # This measures how much the representation changes over time
        # In our case, we can use the difference between output and target as a proxy for temporal change
        if self.config.sim_coeff_t > 0:
            sim_loss_t = torch.mean((output_proj - target_proj).pow(2))
        else:
            sim_loss_t = torch.tensor(0.0, device=output.device)
        
        # For temporal std/variance loss (std_coeff_t)
        # This regularizes the standard deviation across temporal dimensions
        # Since we typically process batches, we can treat the batch as a proxy for temporal dimension
        if self.config.std_coeff_t > 0:
            # Stack output and target to create a "temporal" dimension
            temporal_stack = torch.stack([output_proj, target_proj], dim=1)  # [batch_size, 2, feature_dim]
            std_loss_t = self._std_loss(temporal_stack, across_time=True)
        else:
            std_loss_t = torch.tensor(0.0, device=output.device)
        
        # For temporal covariance loss (cov_coeff_t)
        if self.config.cov_coeff_t > 0:
            # Same temporal stack for covariance
            temporal_stack = torch.stack([output_proj, target_proj], dim=1)  # [batch_size, 2, feature_dim]
            cov_loss_t = self._cov_loss(temporal_stack, across_time=True)
        else:
            cov_loss_t = torch.tensor(0.0, device=output.device)
        
        # Compute total loss
        total_loss = (
            self.config.sim_coeff * sim_loss +
            self.config.std_coeff * std_loss +
            self.config.cov_coeff * cov_loss +
            self.config.sim_coeff_t * sim_loss_t +
            self.config.std_coeff_t * std_loss_t +
            self.config.cov_coeff_t * cov_loss_t
        )
        
        return VICRegLossInfo(
            total_loss=total_loss,
            sim_loss=sim_loss,
            std_loss=std_loss,
            cov_loss=cov_loss,
            std_loss_t=std_loss_t,
            cov_loss_t=cov_loss_t,
            sim_loss_t=sim_loss_t,
            loss_name=f"vicreg_{self.model_type}",
            name_prefix=self.name_prefix
        )
    
    def _std_loss(self, x: torch.Tensor, across_time: bool = False) -> torch.Tensor:
        """
        Compute the standard deviation loss.
        Encourages the standard deviation of each feature to be at least std_margin.
        
        Args:
            x: Representations of shape [batch_size, feature_dim] or [batch_size, time, feature_dim]
            across_time: Whether to compute loss across time
            
        Returns:
            Standard deviation loss
        """
        # If computing across time, the expected shape is [batch_size, time_steps, features]
        # Otherwise, the expected shape is [batch_size, features]
        
        # Center the representations
        if across_time:
            # Mean across time dimension (dim=1)
            x = x - x.mean(dim=1, keepdim=True)
            
            # Only compute if std_coeff_t is non-zero
            if self.config.std_coeff_t > 0:
                # Compute variance across time for each feature
                std = torch.sqrt(x.var(dim=1) + 1e-4)  # [batch_size, features]
                # Apply margin using the temporal margin
                std_loss = F.relu(self.config.std_margin_t - std).mean()
            else:
                std_loss = torch.tensor(0.0, device=x.device)
        else:
            # Mean across batch dimension (dim=0)
            x = x - x.mean(dim=0, keepdim=True)
            
            # Only compute if std_coeff is non-zero
            if self.config.std_coeff > 0:
                # Compute variance across batch for each feature
                std = torch.sqrt(x.var(dim=0) + 1e-4)  # [features]
                # Apply margin
                std_loss = F.relu(self.config.std_margin - std).mean()
            else:
                std_loss = torch.tensor(0.0, device=x.device)
        
        return std_loss
    
    def _cov_loss(self, x: torch.Tensor, across_time: bool = False) -> torch.Tensor:
        """
        Compute the covariance loss.
        Pushes the covariance matrix (except diagonal) to be zero.
        
        Args:
            x: Representations of shape [batch_size, feature_dim] or [batch_size, time, feature_dim]
            across_time: Whether to compute loss across time
            
        Returns:
            Covariance loss
        """
        if across_time:
            # For temporal covariance, we compute covariance across time dimension
            # Expected shape: [batch_size, time_steps, features]
            if self.config.cov_coeff_t <= 0:
                return torch.tensor(0.0, device=x.device)
            
            batch_size, time_steps, feature_dim = x.shape
            
            # Center the representations across time dimension
            x = x - x.mean(dim=1, keepdim=True)
            
            # Compute covariance for each batch sample
            cov_losses = []
            for b in range(batch_size):
                # Compute covariance matrix [features x features] for this batch sample
                # across time dimension
                sample_x = x[b]  # [time_steps, features]
                cov = (sample_x.T @ sample_x) / (time_steps - 1)  # [features, features]
                
                # Remove diagonal elements
                cov_diag = torch.diag(cov)
                cov = cov - torch.diag(cov_diag)
                
                # Compute Frobenius norm of off-diagonal elements
                cov_loss = (cov ** 2).sum() / feature_dim
                if self.config.adjust_cov:
                    cov_loss = cov_loss / (feature_dim - 1)  # Normalize by number of off-diagonal elements
                
                cov_losses.append(cov_loss)
            
            # Average across batch
            cov_loss = torch.stack(cov_losses).mean()
        else:
            # Standard covariance loss across batch dimension
            if self.config.cov_coeff <= 0:
                return torch.tensor(0.0, device=x.device)
                
            batch_size, feature_dim = x.shape
            
            # Center the representations
            x = x - x.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = (x.T @ x) / (batch_size - 1)  # [feature_dim, feature_dim]
            
            # Zero out the diagonal
            cov_diag = torch.diag(cov)
            cov = cov - torch.diag(cov_diag)
            
            # Compute Frobenius norm of the off-diagonal elements
            cov_loss = (cov ** 2).sum() / feature_dim
            if self.config.adjust_cov:
                cov_loss = cov_loss / (feature_dim - 1)  # Normalize by number of off-diagonal elements
        
        return cov_loss


def get_loss_function(
    loss_type: str,
    model_type: str,
    input_dim: int = None,
    config: Dict[str, Any] = None,
    name_prefix: str = ""
) -> nn.Module:
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type: Type of loss function ('mse' or 'vicreg')
        model_type: Type of model ('dynamics' or 'reward')
        input_dim: Input dimension for the loss function (required for 'vicreg')
        config: Configuration for the loss function (required for 'vicreg')
        name_prefix: Prefix for logging names
        
    Returns:
        Loss function module
    """
    if loss_type.lower() == 'mse':
        return MSELoss(name_prefix=name_prefix, model_type=model_type)
    
    elif loss_type.lower() == 'vicreg':
        if input_dim is None:
            raise ValueError("input_dim must be provided for VICReg loss")
        
        if config is None:
            logger.warning("No configuration provided for VICReg, using defaults")
            vicreg_config = VICRegConfig()
        else:
            # Convert dict to VICRegConfig
            vicreg_config = VICRegConfig()
            for key, value in config.items():
                if hasattr(vicreg_config, key):
                    setattr(vicreg_config, key, value)
        
        return VICRegLoss(
            config=vicreg_config,
            input_dim=input_dim,
            name_prefix=name_prefix,
            model_type=model_type
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 