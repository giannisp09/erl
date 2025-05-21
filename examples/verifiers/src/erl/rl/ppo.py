from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

class PPOPolicy(nn.Module):
    """Policy network for PPO."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(num_layers - 1)],
        )
        
        # Policy head
        self.policy_mean = nn.Linear(hidden_dim, action_dim)
        self.policy_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        observation: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, Dict[str, Any]]:
        features = self.feature_extractor(observation)
        
        # Policy
        mean = self.policy_mean(features)
        std = torch.exp(self.policy_logstd)
        dist = Independent(Normal(mean, std), 1)
        
        # Value
        value = self.value_head(features).squeeze(-1)
        
        # Action log probability if action is provided
        log_prob = None
        if action is not None:
            log_prob = dist.log_prob(action)
            
        return dist, value, {"log_prob": log_prob}

class IntrinsicCuriosityModule(nn.Module):
    """Intrinsic Curiosity Module for exploration."""
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Forward model
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Inverse model
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        # Encode observations
        phi = self.encoder(observation)
        phi_next = self.encoder(next_observation)
        
        # Forward model prediction
        phi_next_pred = self.forward_model(torch.cat([phi, action], dim=-1))
        forward_loss = F.mse_loss(phi_next_pred, phi_next)
        
        # Inverse model prediction
        pred_action = self.inverse_model(torch.cat([phi, phi_next], dim=-1))
        inverse_loss = F.mse_loss(pred_action, action)
        
        # Intrinsic reward is the forward model error
        intrinsic_reward = forward_loss.detach()
        
        return intrinsic_reward, forward_loss + inverse_loss, {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
        }

class PPO:
    """Proximal Policy Optimization with Intrinsic Curiosity."""
    
    def __init__(
        self,
        policy: PPOPolicy,
        curiosity: IntrinsicCuriosityModule,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        num_epochs: int = 10,
        intrinsic_weight: float = 0.1,
    ):
        self.policy = policy
        self.curiosity = curiosity
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.curiosity_optimizer = torch.optim.Adam(curiosity.parameters(), lr=learning_rate)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.num_epochs = num_epochs
        self.intrinsic_weight = intrinsic_weight
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            
        return advantages
        
    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """Update policy and curiosity module."""
        # Compute intrinsic rewards and curiosity loss
        intrinsic_rewards, curiosity_loss, curiosity_info = self.curiosity(
            observations, actions, next_observations,
        )
        
        # Combine extrinsic and intrinsic rewards
        total_rewards = rewards + self.intrinsic_weight * intrinsic_rewards
        
        # Get policy values and distributions
        _, values, _ = self.policy(observations)
        dist, _, _ = self.policy(observations, actions)
        log_probs = dist.log_prob(actions)
        
        # Compute advantages
        advantages = self.compute_gae(total_rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.num_epochs):
            # Compute policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            policy_loss1 = ratio * advantages
            policy_loss2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, total_rewards)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss
            
            # Update policy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update curiosity
            self.curiosity_optimizer.zero_grad()
            curiosity_loss.backward()
            self.curiosity_optimizer.step()
            
            # Early stopping if KL divergence is too large
            with torch.no_grad():
                kl = (old_log_probs - log_probs).mean()
                if kl > self.target_kl:
                    break
                    
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "curiosity_loss": curiosity_loss.item(),
            "kl_divergence": kl.item(),
            **curiosity_info,
        } 