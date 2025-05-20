import pytest
import torch
import numpy as np
from src.rewards.grounded import (
    EnvironmentalReward,
    DistanceReward,
    TaskCompletionReward,
    UserFeedbackReward,
    SafetyReward,
)

def test_distance_reward():
    """Test distance-based reward function."""
    target = torch.tensor([1.0, 1.0])
    reward_fn = DistanceReward(target=target)
    
    # Test at target
    observation = torch.tensor([1.0, 1.0])
    action = torch.tensor([0.0, 0.0])
    next_observation = torch.tensor([1.0, 1.0])
    reward, info = reward_fn.compute(observation, action, next_observation)
    assert reward == 0.0
    assert "raw_reward" in info
    
    # Test away from target
    next_observation = torch.tensor([2.0, 2.0])
    reward, info = reward_fn.compute(observation, action, next_observation)
    assert reward < 0.0
    assert "raw_reward" in info

def test_task_completion_reward():
    """Test task completion reward function."""
    reward_fn = TaskCompletionReward(success_threshold=0.9)
    
    # Test successful completion
    observation = torch.tensor([0.0])
    action = torch.tensor([0.0])
    next_observation = torch.tensor([0.0])
    info = {"success": True}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == 1.0
    assert "raw_reward" in info
    
    # Test unsuccessful completion
    info = {"success": False}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == 0.0
    assert "raw_reward" in info

def test_user_feedback_reward():
    """Test user feedback reward function."""
    reward_fn = UserFeedbackReward(feedback_key="user_feedback")
    
    # Test positive feedback
    observation = torch.tensor([0.0])
    action = torch.tensor([0.0])
    next_observation = torch.tensor([0.0])
    info = {"user_feedback": 1.0}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == 1.0
    assert "raw_reward" in info
    
    # Test negative feedback
    info = {"user_feedback": -1.0}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == -1.0
    assert "raw_reward" in info
    
    # Test boolean feedback
    info = {"user_feedback": True}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == 1.0
    assert "raw_reward" in info

def test_safety_reward():
    """Test safety reward function."""
    reward_fn = SafetyReward(safety_threshold=0.5, penalty=-1.0)
    
    # Test safe state
    observation = torch.tensor([0.0])
    action = torch.tensor([0.0])
    next_observation = torch.tensor([0.0])
    info = {"safety": 0.8}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == 0.0
    assert "raw_reward" in info
    
    # Test unsafe state
    info = {"safety": 0.3}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == -1.0
    assert "raw_reward" in info
    
    # Test boolean safety
    info = {"safety": True}
    reward, info = reward_fn.compute(observation, action, next_observation, info)
    assert reward == 0.0
    assert "raw_reward" in info

def test_reward_scaling():
    """Test reward scaling and clipping."""
    reward_fn = DistanceReward(
        target=torch.tensor([1.0, 1.0]),
        weight=2.0,
        scale=0.5,
        clip_range=(-1.0, 1.0),
    )
    
    # Test scaling
    observation = torch.tensor([0.0, 0.0])
    action = torch.tensor([0.0, 0.0])
    next_observation = torch.tensor([2.0, 2.0])
    reward, info = reward_fn.compute(observation, action, next_observation)
    assert reward == -1.0  # Clipped to -1.0
    assert "raw_reward" in info

def test_reward_combination():
    """Test combining multiple reward functions."""
    distance_reward = DistanceReward(
        target=torch.tensor([1.0, 1.0]),
        weight=1.0,
    )
    safety_reward = SafetyReward(
        safety_threshold=0.5,
        penalty=-1.0,
        weight=0.5,
    )
    
    observation = torch.tensor([0.0, 0.0])
    action = torch.tensor([0.0, 0.0])
    next_observation = torch.tensor([2.0, 2.0])
    info = {"safety": 0.3}
    
    # Compute individual rewards
    distance_r, _ = distance_reward.compute(observation, action, next_observation)
    safety_r, _ = safety_reward.compute(observation, action, next_observation, info)
    
    # Combined reward should be weighted sum
    expected_reward = distance_r + 0.5 * safety_r
    assert expected_reward < 0.0 