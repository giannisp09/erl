from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch

class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def compute(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the reward for a given transition.
        
        Args:
            observation: Current observation
            action: Action taken
            next_observation: Next observation
            info: Additional information about the transition
            
        Returns:
            Tuple of (reward, info) where info contains additional metrics
        """
        pass
    
    @abstractmethod
    def update(self, experience: Any) -> None:
        """
        Update the reward function based on new experience.
        
        Args:
            experience: Experience tuple or batch
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the reward function to its initial state."""
        pass

class CompositeReward(RewardFunction):
    """Combines multiple reward functions with weights."""
    
    def __init__(
        self,
        reward_functions: Dict[str, Tuple[RewardFunction, float]],
    ):
        """
        Args:
            reward_functions: Dictionary mapping reward names to (function, weight) tuples
        """
        self.reward_functions = reward_functions
        
    def compute(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        total_reward = 0.0
        reward_info = {}
        
        for name, (reward_fn, weight) in self.reward_functions.items():
            reward, info = reward_fn.compute(observation, action, next_observation, info)
            total_reward += weight * reward
            reward_info[f"{name}_reward"] = reward
            
        reward_info["total_reward"] = total_reward
        return total_reward, reward_info
        
    def update(self, experience: Any) -> None:
        for reward_fn, _ in self.reward_functions.values():
            reward_fn.update(experience)
            
    def reset(self) -> None:
        for reward_fn, _ in self.reward_functions.values():
            reward_fn.reset() 