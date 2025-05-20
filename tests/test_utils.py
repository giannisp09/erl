import pytest
import torch
import numpy as np
from src.utils.metrics import (
    compute_gae,
    compute_returns,
    compute_entropy,
    compute_kl_divergence,
    MetricsTracker,
)

def test_compute_gae():
    """Test Generalized Advantage Estimation computation."""
    # Create dummy data
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
    dones = torch.tensor([0, 0, 0, 1])
    gamma = 0.99
    lambda_ = 0.95
    
    # Compute GAE
    advantages = compute_gae(
        rewards=rewards,
        values=values,
        next_values=next_values,
        dones=dones,
        gamma=gamma,
        lambda_=lambda_,
    )
    
    assert advantages.shape == rewards.shape
    assert not torch.isnan(advantages).any()
    assert not torch.isinf(advantages).any()

def test_compute_returns():
    """Test discounted returns computation."""
    # Create dummy data
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    dones = torch.tensor([0, 0, 0, 1])
    gamma = 0.99
    
    # Compute returns
    returns = compute_returns(
        rewards=rewards,
        dones=dones,
        gamma=gamma,
    )
    
    assert returns.shape == rewards.shape
    assert not torch.isnan(returns).any()
    assert not torch.isinf(returns).any()
    assert returns[-1] == rewards[-1]  # Last return should equal last reward

def test_compute_entropy():
    """Test entropy computation."""
    # Create dummy logits
    logits = torch.tensor([
        [0.0, 0.0],  # Uniform distribution
        [1.0, 0.0],  # Deterministic distribution
    ])
    
    # Compute entropy
    entropy = compute_entropy(logits)
    
    assert entropy.shape == (2,)
    assert not torch.isnan(entropy).any()
    assert not torch.isinf(entropy).any()
    assert entropy[0] > entropy[1]  # Uniform should have higher entropy

def test_compute_kl_divergence():
    """Test KL divergence computation."""
    # Create dummy logits
    p_logits = torch.tensor([
        [0.0, 0.0],  # Uniform distribution
        [1.0, 0.0],  # Deterministic distribution
    ])
    q_logits = torch.tensor([
        [0.0, 0.0],  # Same as p
        [0.0, 1.0],  # Different from p
    ])
    
    # Compute KL divergence
    kl = compute_kl_divergence(p_logits, q_logits)
    
    assert kl.shape == (2,)
    assert not torch.isnan(kl).any()
    assert not torch.isinf(kl).any()
    assert kl[0] == 0.0  # Same distributions should have zero KL
    assert kl[1] > 0.0  # Different distributions should have positive KL

def test_metrics_tracker_initialization():
    """Test metrics tracker initialization."""
    tracker = MetricsTracker()
    assert isinstance(tracker.metrics, dict)
    assert isinstance(tracker.episode_metrics, dict)

def test_metrics_tracker_update():
    """Test metrics tracker update."""
    tracker = MetricsTracker()
    
    # Update metrics
    tracker.update("loss", 0.5)
    tracker.update("reward", 1.0)
    tracker.update("entropy", 0.2)
    
    # Check metrics
    assert "loss" in tracker.metrics
    assert "reward" in tracker.metrics
    assert "entropy" in tracker.metrics
    assert len(tracker.metrics["loss"]) == 1
    assert len(tracker.metrics["reward"]) == 1
    assert len(tracker.metrics["entropy"]) == 1

def test_metrics_tracker_get_metrics():
    """Test metrics tracker get_metrics."""
    tracker = MetricsTracker()
    
    # Update metrics multiple times
    for i in range(5):
        tracker.update("loss", float(i))
        tracker.update("reward", float(i))
    
    # Get metrics with different windows
    metrics_1 = tracker.get_metrics(window=1)
    metrics_3 = tracker.get_metrics(window=3)
    metrics_5 = tracker.get_metrics(window=5)
    
    assert "loss" in metrics_1
    assert "reward" in metrics_1
    assert metrics_1["loss"] == 4.0  # Last value
    assert metrics_1["reward"] == 4.0  # Last value
    
    assert metrics_3["loss"] == pytest.approx(3.0)  # Average of last 3
    assert metrics_3["reward"] == pytest.approx(3.0)  # Average of last 3
    
    assert metrics_5["loss"] == pytest.approx(2.0)  # Average of all 5
    assert metrics_5["reward"] == pytest.approx(2.0)  # Average of all 5

def test_metrics_tracker_episode_metrics():
    """Test metrics tracker episode metrics."""
    tracker = MetricsTracker()
    
    # Update episode metrics
    tracker.update("episode_reward", 10.0, is_episode=True)
    tracker.update("episode_length", 100, is_episode=True)
    
    # Check episode metrics
    assert "episode_reward" in tracker.episode_metrics
    assert "episode_length" in tracker.episode_metrics
    assert len(tracker.episode_metrics["episode_reward"]) == 1
    assert len(tracker.episode_metrics["episode_length"]) == 1

def test_metrics_tracker_get_episode_metrics():
    """Test metrics tracker get_episode_metrics."""
    tracker = MetricsTracker()
    
    # Update episode metrics multiple times
    for i in range(5):
        tracker.update("episode_reward", float(i), is_episode=True)
        tracker.update("episode_length", i * 10, is_episode=True)
    
    # Get episode metrics with different windows
    metrics_1 = tracker.get_episode_metrics(window=1)
    metrics_3 = tracker.get_episode_metrics(window=3)
    metrics_5 = tracker.get_episode_metrics(window=5)
    
    assert "episode_reward" in metrics_1
    assert "episode_length" in metrics_1
    assert metrics_1["episode_reward"] == 4.0  # Last value
    assert metrics_1["episode_length"] == 40  # Last value
    
    assert metrics_3["episode_reward"] == pytest.approx(3.0)  # Average of last 3
    assert metrics_3["episode_length"] == pytest.approx(30.0)  # Average of last 3
    
    assert metrics_5["episode_reward"] == pytest.approx(2.0)  # Average of all 5
    assert metrics_5["episode_length"] == pytest.approx(20.0)  # Average of all 5

def test_metrics_tracker_reset():
    """Test metrics tracker reset."""
    tracker = MetricsTracker()
    
    # Update metrics
    tracker.update("loss", 0.5)
    tracker.update("reward", 1.0)
    tracker.update("episode_reward", 10.0, is_episode=True)
    
    # Reset metrics
    tracker.reset()
    
    # Check metrics are empty
    assert len(tracker.metrics) == 0
    assert len(tracker.episode_metrics) == 0 