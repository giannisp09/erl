import pytest
import torch
import numpy as np
from src.rl.ppo import PPO
from src.rl.curiosity import IntrinsicCuriosityModule
from src.rl.temporal_abstraction import TemporalAbstraction

def test_ppo_initialization():
    """Test PPO initialization."""
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    ppo = PPO(
        policy=policy,
        clip_ratio=0.2,
        target_kl=0.01,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    assert ppo.clip_ratio == 0.2
    assert ppo.target_kl == 0.01
    assert ppo.value_coef == 0.5
    assert ppo.entropy_coef == 0.01
    assert ppo.max_grad_norm == 0.5

def test_ppo_update():
    """Test PPO update step."""
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    ppo = PPO(
        policy=policy,
        clip_ratio=0.2,
        target_kl=0.01,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    
    # Create dummy batch
    batch_size = 32
    observations = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    rewards = torch.randn(batch_size)
    next_observations = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)
    
    # Update policy
    loss = ppo.update(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
    )
    assert isinstance(loss, float)
    assert loss >= 0.0

def test_curiosity_initialization():
    """Test Intrinsic Curiosity Module initialization."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    curiosity = IntrinsicCuriosityModule(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        beta=0.2,
        learning_rate=1e-4,
    )
    assert curiosity.state_dim == state_dim
    assert curiosity.action_dim == action_dim
    assert curiosity.hidden_size == hidden_size
    assert curiosity.beta == 0.2

def test_curiosity_update():
    """Test Intrinsic Curiosity Module update."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    curiosity = IntrinsicCuriosityModule(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        beta=0.2,
        learning_rate=1e-4,
    )
    
    # Create dummy batch
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    next_states = torch.randn(batch_size, state_dim)
    
    # Update curiosity module
    forward_loss, inverse_loss = curiosity.update(
        states=states,
        actions=actions,
        next_states=next_states,
    )
    assert isinstance(forward_loss, float)
    assert isinstance(inverse_loss, float)
    assert forward_loss >= 0.0
    assert inverse_loss >= 0.0

def test_temporal_abstraction_initialization():
    """Test Temporal Abstraction initialization."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    abstraction = TemporalAbstraction(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_options=4,
        learning_rate=1e-4,
    )
    assert abstraction.state_dim == state_dim
    assert abstraction.action_dim == action_dim
    assert abstraction.hidden_size == hidden_size
    assert abstraction.num_options == 4

def test_temporal_abstraction_update():
    """Test Temporal Abstraction update."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    abstraction = TemporalAbstraction(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_options=4,
        learning_rate=1e-4,
    )
    
    # Create dummy batch
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    next_states = torch.randn(batch_size, state_dim)
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)
    
    # Update temporal abstraction
    option_loss, termination_loss = abstraction.update(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        dones=dones,
    )
    assert isinstance(option_loss, float)
    assert isinstance(termination_loss, float)
    assert option_loss >= 0.0
    assert termination_loss >= 0.0

def test_ppo_with_curiosity():
    """Test PPO with Intrinsic Curiosity Module."""
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    ppo = PPO(
        policy=policy,
        clip_ratio=0.2,
        target_kl=0.01,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    )
    
    curiosity = IntrinsicCuriosityModule(
        state_dim=4,
        action_dim=2,
        hidden_size=64,
        beta=0.2,
        learning_rate=1e-4,
    )
    
    # Create dummy batch
    batch_size = 32
    observations = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    rewards = torch.randn(batch_size)
    next_observations = torch.randn(batch_size, 4)
    dones = torch.zeros(batch_size)
    
    # Update curiosity module
    forward_loss, inverse_loss = curiosity.update(
        states=observations,
        actions=actions,
        next_states=next_observations,
    )
    
    # Add intrinsic reward
    intrinsic_reward = curiosity.get_intrinsic_reward(
        states=observations,
        actions=actions,
        next_states=next_observations,
    )
    total_rewards = rewards + 0.1 * intrinsic_reward
    
    # Update policy with combined rewards
    loss = ppo.update(
        observations=observations,
        actions=actions,
        rewards=total_rewards,
        next_observations=next_observations,
        dones=dones,
    )
    assert isinstance(loss, float)
    assert loss >= 0.0 