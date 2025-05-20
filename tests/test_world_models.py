import pytest
import torch
import numpy as np
from src.models.world_models import (
    DeterministicWorldModel,
    StochasticWorldModel,
    EnsembleWorldModel,
    LatentWorldModel,
)

def test_deterministic_world_model_initialization():
    """Test deterministic world model initialization."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    model = DeterministicWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
    )
    assert model.state_dim == state_dim
    assert model.action_dim == action_dim
    assert model.hidden_size == hidden_size

def test_deterministic_world_model_forward():
    """Test deterministic world model forward pass."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    model = DeterministicWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
    )
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    next_states = model(states, actions)
    assert next_states.shape == (batch_size, state_dim)

def test_stochastic_world_model_initialization():
    """Test stochastic world model initialization."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    latent_dim = 16
    
    model = StochasticWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
    )
    assert model.state_dim == state_dim
    assert model.action_dim == action_dim
    assert model.hidden_size == hidden_size
    assert model.latent_dim == latent_dim

def test_stochastic_world_model_forward():
    """Test stochastic world model forward pass."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    latent_dim = 16
    
    model = StochasticWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
    )
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    next_states, log_prob = model(states, actions)
    assert next_states.shape == (batch_size, state_dim)
    assert log_prob.shape == (batch_size,)

def test_ensemble_world_model_initialization():
    """Test ensemble world model initialization."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    num_models = 5
    
    model = EnsembleWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_models=num_models,
    )
    assert model.state_dim == state_dim
    assert model.action_dim == action_dim
    assert model.hidden_size == hidden_size
    assert model.num_models == num_models

def test_ensemble_world_model_forward():
    """Test ensemble world model forward pass."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    num_models = 5
    
    model = EnsembleWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_models=num_models,
    )
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    next_states, uncertainties = model(states, actions)
    assert next_states.shape == (batch_size, state_dim)
    assert uncertainties.shape == (batch_size,)

def test_latent_world_model_initialization():
    """Test latent world model initialization."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    latent_dim = 16
    
    model = LatentWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
    )
    assert model.state_dim == state_dim
    assert model.action_dim == action_dim
    assert model.hidden_size == hidden_size
    assert model.latent_dim == latent_dim

def test_latent_world_model_forward():
    """Test latent world model forward pass."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    latent_dim = 16
    
    model = LatentWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
    )
    
    # Test forward pass
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    next_states, latents = model(states, actions)
    assert next_states.shape == (batch_size, state_dim)
    assert latents.shape == (batch_size, latent_dim)

def test_world_model_training():
    """Test training world models."""
    state_dim = 4
    action_dim = 2
    hidden_size = 64
    
    # Initialize models
    deterministic = DeterministicWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
    )
    stochastic = StochasticWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=16,
    )
    ensemble = EnsembleWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_models=5,
    )
    latent = LatentWorldModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=16,
    )
    
    # Create dummy data
    batch_size = 32
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    next_states = torch.randn(batch_size, state_dim)
    
    # Train deterministic model
    deterministic_optimizer = torch.optim.Adam(deterministic.parameters())
    deterministic_optimizer.zero_grad()
    pred_next_states = deterministic(states, actions)
    loss = torch.nn.functional.mse_loss(pred_next_states, next_states)
    loss.backward()
    deterministic_optimizer.step()
    assert loss.item() > 0.0
    
    # Train stochastic model
    stochastic_optimizer = torch.optim.Adam(stochastic.parameters())
    stochastic_optimizer.zero_grad()
    pred_next_states, log_prob = stochastic(states, actions)
    loss = -log_prob.mean() + torch.nn.functional.mse_loss(pred_next_states, next_states)
    loss.backward()
    stochastic_optimizer.step()
    assert loss.item() > 0.0
    
    # Train ensemble model
    ensemble_optimizer = torch.optim.Adam(ensemble.parameters())
    ensemble_optimizer.zero_grad()
    pred_next_states, uncertainties = ensemble(states, actions)
    loss = torch.nn.functional.mse_loss(pred_next_states, next_states) + uncertainties.mean()
    loss.backward()
    ensemble_optimizer.step()
    assert loss.item() > 0.0
    
    # Train latent model
    latent_optimizer = torch.optim.Adam(latent.parameters())
    latent_optimizer.zero_grad()
    pred_next_states, latents = latent(states, actions)
    loss = torch.nn.functional.mse_loss(pred_next_states, next_states) + latents.pow(2).mean()
    loss.backward()
    latent_optimizer.step()
    assert loss.item() > 0.0 