from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np
from .base import RewardFunction

class EnvironmentalReward(RewardFunction):
    """Reward function that pulls signals from the environment."""
    
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        scale: float = 1.0,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.name = name
        self.weight = weight
        self.scale = scale
        self.clip_range = clip_range
        
    def compute(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward from environmental signals."""
        reward = self._get_environmental_signal(observation, action, next_observation, info)
        
        # Scale and clip reward
        reward = reward * self.scale
        if self.clip_range is not None:
            reward = np.clip(reward, self.clip_range[0], self.clip_range[1])
            
        return reward * self.weight, {"raw_reward": reward}
        
    @abstractmethod
    def _get_environmental_signal(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Get raw environmental signal."""
        pass

class DistanceReward(EnvironmentalReward):
    """Reward based on distance to target."""
    
    def __init__(
        self,
        target: torch.Tensor,
        distance_metric: str = "euclidean",
        **kwargs,
    ):
        super().__init__(name="distance", **kwargs)
        self.target = target
        self.distance_metric = distance_metric
        
    def _get_environmental_signal(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        if self.distance_metric == "euclidean":
            distance = torch.norm(next_observation - self.target)
        elif self.distance_metric == "manhattan":
            distance = torch.sum(torch.abs(next_observation - self.target))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        return -distance.item()

class TaskCompletionReward(EnvironmentalReward):
    """Reward based on task completion."""
    
    def __init__(
        self,
        success_threshold: float = 0.9,
        **kwargs,
    ):
        super().__init__(name="task_completion", **kwargs)
        self.success_threshold = success_threshold
        
    def _get_environmental_signal(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        if info is None or "success" not in info:
            return 0.0
            
        success = info["success"]
        if isinstance(success, bool):
            return 1.0 if success else 0.0
        else:
            return float(success)

class UserFeedbackReward(EnvironmentalReward):
    """Reward based on user feedback."""
    
    def __init__(
        self,
        feedback_key: str = "user_feedback",
        **kwargs,
    ):
        super().__init__(name="user_feedback", **kwargs)
        self.feedback_key = feedback_key
        
    def _get_environmental_signal(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        if info is None or self.feedback_key not in info:
            return 0.0
            
        feedback = info[self.feedback_key]
        if isinstance(feedback, (int, float)):
            return float(feedback)
        elif isinstance(feedback, bool):
            return 1.0 if feedback else -1.0
        else:
            return 0.0

class SafetyReward(EnvironmentalReward):
    """Reward based on safety constraints."""
    
    def __init__(
        self,
        safety_threshold: float = 0.5,
        penalty: float = -1.0,
        **kwargs,
    ):
        super().__init__(name="safety", **kwargs)
        self.safety_threshold = safety_threshold
        self.penalty = penalty
        
    def _get_environmental_signal(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        if info is None or "safety" not in info:
            return 0.0
            
        safety = info["safety"]
        if isinstance(safety, bool):
            return 0.0 if safety else self.penalty
        else:
            return 0.0 if safety >= self.safety_threshold else self.penalty 