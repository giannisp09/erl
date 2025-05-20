import pytest
import torch
import numpy as np
from src.planning.planner import MCTSPlanner
from src.planning.search import BeamSearch, MPC, CEM

def test_mcts_planner_initialization():
    """Test MCTS planner initialization."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    
    planner = MCTSPlanner(
        world_model=world_model,
        policy=policy,
        horizon=10,
        num_simulations=100,
        exploration_constant=1.0,
        temperature=1.0,
    )
    assert planner.horizon == 10
    assert planner.num_simulations == 100
    assert planner.exploration_constant == 1.0
    assert planner.temperature == 1.0

def test_mcts_planner_plan():
    """Test MCTS planner planning."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    
    planner = MCTSPlanner(
        world_model=world_model,
        policy=policy,
        horizon=10,
        num_simulations=100,
        exploration_constant=1.0,
        temperature=1.0,
    )
    
    # Test planning
    state = torch.randn(4)
    action, info = planner.plan(state)
    assert action.shape == (2,)
    assert "value" in info
    assert "visit_counts" in info
    assert "action_probs" in info

def test_beam_search_initialization():
    """Test Beam Search initialization."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    
    search = BeamSearch(
        world_model=world_model,
        beam_size=5,
        horizon=10,
    )
    assert search.beam_size == 5
    assert search.horizon == 10

def test_beam_search_plan():
    """Test Beam Search planning."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    
    search = BeamSearch(
        world_model=world_model,
        beam_size=5,
        horizon=10,
    )
    
    # Test planning
    state = torch.randn(4)
    action_sequence, info = search.plan(state)
    assert len(action_sequence) == 10
    assert "values" in info
    assert "rewards" in info

def test_mpc_initialization():
    """Test Model Predictive Control initialization."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    
    mpc = MPC(
        world_model=world_model,
        num_samples=100,
        horizon=10,
    )
    assert mpc.num_samples == 100
    assert mpc.horizon == 10

def test_mpc_plan():
    """Test Model Predictive Control planning."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    
    mpc = MPC(
        world_model=world_model,
        num_samples=100,
        horizon=10,
    )
    
    # Test planning
    state = torch.randn(4)
    action_sequence, info = mpc.plan(state)
    assert len(action_sequence) == 10
    assert "values" in info
    assert "rewards" in info

def test_cem_initialization():
    """Test Cross-Entropy Method initialization."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    
    cem = CEM(
        world_model=world_model,
        num_samples=100,
        num_elites=10,
        num_iterations=5,
        horizon=10,
    )
    assert cem.num_samples == 100
    assert cem.num_elites == 10
    assert cem.num_iterations == 5
    assert cem.horizon == 10

def test_cem_plan():
    """Test Cross-Entropy Method planning."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    
    cem = CEM(
        world_model=world_model,
        num_samples=100,
        num_elites=10,
        num_iterations=5,
        horizon=10,
    )
    
    # Test planning
    state = torch.randn(4)
    action_sequence, info = cem.plan(state)
    assert len(action_sequence) == 10
    assert "values" in info
    assert "rewards" in info
    assert "elite_actions" in info

def test_planning_comparison():
    """Test comparison of different planning methods."""
    world_model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    )
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    
    # Initialize planners
    mcts = MCTSPlanner(
        world_model=world_model,
        policy=policy,
        horizon=10,
        num_simulations=100,
    )
    beam_search = BeamSearch(
        world_model=world_model,
        beam_size=5,
        horizon=10,
    )
    mpc = MPC(
        world_model=world_model,
        num_samples=100,
        horizon=10,
    )
    cem = CEM(
        world_model=world_model,
        num_samples=100,
        num_elites=10,
        num_iterations=5,
        horizon=10,
    )
    
    # Test planning with same state
    state = torch.randn(4)
    
    # Get plans from each method
    mcts_action, mcts_info = mcts.plan(state)
    beam_sequence, beam_info = beam_search.plan(state)
    mpc_sequence, mpc_info = mpc.plan(state)
    cem_sequence, cem_info = cem.plan(state)
    
    # Compare values
    mcts_value = mcts_info["value"]
    beam_value = beam_info["values"][0]
    mpc_value = mpc_info["values"][0]
    cem_value = cem_info["values"][0]
    
    # All methods should find reasonable plans
    assert mcts_value > -np.inf
    assert beam_value > -np.inf
    assert mpc_value > -np.inf
    assert cem_value > -np.inf 