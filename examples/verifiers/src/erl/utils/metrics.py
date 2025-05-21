from typing import Dict, List, Optional
import numpy as np
import torch
from collections import defaultdict

class MetricsTracker:
    """Track and compute various metrics during training."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.current_episode = 0
    
    def update(
        self,
        metrics: Dict[str, float],
        is_episode_end: bool = False,
    ) -> None:
        """Update metrics with new values."""
        for k, v in metrics.items():
            self.metrics[k].append(v)
            if is_episode_end:
                self.episode_metrics[k].append(v)
    
    def get_metrics(
        self,
        window: Optional[int] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Get computed metrics over a window of steps."""
        result = {}
        for k, v in self.metrics.items():
            if window is not None:
                v = v[-window:]
            if len(v) > 0:
                result[f"{prefix}{k}"] = np.mean(v)
        return result
    
    def get_episode_metrics(
        self,
        window: Optional[int] = None,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Get computed metrics over a window of episodes."""
        result = {}
        for k, v in self.episode_metrics.items():
            if window is not None:
                v = v[-window:]
            if len(v) > 0:
                result[f"{prefix}{k}"] = np.mean(v)
        return result
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.episode_metrics.clear()
        self.current_episode = 0

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]
        
        delta = (
            rewards[t]
            + gamma * next_value * (1 - dones[t])
            - values[t]
        )
        last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    
    return advantages

def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted returns."""
    returns = torch.zeros_like(rewards)
    last_return = 0
    
    for t in reversed(range(len(rewards))):
        last_return = rewards[t] + gamma * last_return * (1 - dones[t])
        returns[t] = last_return
    
    return returns

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of a categorical distribution."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)

def compute_kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute KL divergence between two categorical distributions."""
    p_probs = torch.softmax(p_logits, dim=-1)
    p_log_probs = torch.log_softmax(p_logits, dim=-1)
    q_log_probs = torch.log_softmax(q_logits, dim=-1)
    return (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1) 