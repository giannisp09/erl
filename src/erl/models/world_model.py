from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

class WorldModel(nn.Module, ABC):
    """Abstract base class for world models."""
    
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
    @abstractmethod
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Predict next state and reward given current state and action.
        
        Args:
            observation: Current observation
            action: Action taken
            hidden_state: Optional hidden state for recurrent models
            
        Returns:
            Tuple of (next_observation, reward, info) where info contains additional metrics
        """
        pass
    
    @abstractmethod
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation into latent state."""
        pass
    
    @abstractmethod
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state into observation."""
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for training the world model.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            reward: Observed reward
            
        Returns:
            Tuple of (loss, info) where info contains additional metrics
        """
        pass

class MLPWorldModel(WorldModel):
    """Simple MLP-based world model."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__(observation_dim, action_dim, hidden_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Dynamics model
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_dim),
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        latent = self.encode(observation)
        next_latent = self.dynamics(torch.cat([latent, action], dim=-1))
        next_observation = self.decode(next_latent)
        reward = self.reward_predictor(next_latent).squeeze(-1)
        return next_observation, reward, {}
        
    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)
        
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
        
    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pred_next_obs, pred_reward, _ = self(observation, action)
        
        obs_loss = nn.functional.mse_loss(pred_next_obs, next_observation)
        reward_loss = nn.functional.mse_loss(pred_reward, reward)
        
        total_loss = obs_loss + reward_loss
        
        return total_loss, {
            "obs_loss": obs_loss.item(),
            "reward_loss": reward_loss.item(),
        } 