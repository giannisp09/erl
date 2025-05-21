from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class Node:
    """Node in the search tree."""
    observation: torch.Tensor
    action: Optional[torch.Tensor] = None
    reward: float = 0.0
    value: float = 0.0
    visit_count: int = 0
    children: Dict[int, 'Node'] = None
    parent: Optional['Node'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}

class Planner(ABC):
    """Abstract base class for planners."""
    
    def __init__(
        self,
        world_model: Any,
        policy: Any,
        horizon: int,
        num_simulations: int,
        exploration_constant: float,
        temperature: float,
    ):
        self.world_model = world_model
        self.policy = policy
        self.horizon = horizon
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
    
    @abstractmethod
    def plan(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan next action given current observation and optional goal."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset planner state."""
        pass

class MCTSPlanner(Planner):
    """Monte Carlo Tree Search planner."""
    
    def __init__(
        self,
        world_model: Any,
        policy: Any,
        horizon: int,
        num_simulations: int,
        exploration_constant: float,
        temperature: float,
    ):
        super().__init__(
            world_model=world_model,
            policy=policy,
            horizon=horizon,
            num_simulations=num_simulations,
            exploration_constant=exploration_constant,
            temperature=temperature,
        )
        self.tree = {}
    
    def plan(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Plan using MCTS."""
        # Initialize root node
        root_key = self._get_state_key(observation)
        if root_key not in self.tree:
            self.tree[root_key] = {
                "visits": 0,
                "value": 0.0,
                "children": {},
            }
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._simulate(observation, goal)
        
        # Select action
        root = self.tree[root_key]
        action_probs = self._get_action_probs(root)
        action = self._sample_action(action_probs)
        
        # Get planning info
        info = {
            "action_probs": action_probs,
            "value": root["value"] / max(1, root["visits"]),
            "visits": root["visits"],
        }
        
        return action, info
    
    def _simulate(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> float:
        """Run one simulation from current state."""
        state = observation
        path = []
        total_reward = 0.0
        
        for _ in range(self.horizon):
            # Select action using UCB
            state_key = self._get_state_key(state)
            if state_key not in self.tree:
                self.tree[state_key] = {
                    "visits": 0,
                    "value": 0.0,
                    "children": {},
                }
            
            node = self.tree[state_key]
            action = self._select_action(node)
            
            # Simulate next state
            next_state, reward = self.world_model.predict(state, action)
            if goal is not None:
                reward += self._goal_reward(next_state, goal)
            
            # Update path
            path.append((state_key, action, reward))
            state = next_state
            total_reward += reward
        
        # Backpropagate
        for state_key, action, reward in reversed(path):
            node = self.tree[state_key]
            node["visits"] += 1
            node["value"] += reward
            if action not in node["children"]:
                node["children"][action] = {"visits": 0, "value": 0.0}
            node["children"][action]["visits"] += 1
            node["children"][action]["value"] += reward
        
        return total_reward
    
    def _select_action(self, node: Dict[str, Any]) -> torch.Tensor:
        """Select action using UCB."""
        if not node["children"]:
            # Use policy for unvisited actions
            return self.policy.act(node["state"])
        
        # Compute UCB scores
        ucb_scores = {}
        for action, child in node["children"].items():
            exploitation = child["value"] / max(1, child["visits"])
            exploration = self.exploration_constant * np.sqrt(
                np.log(max(1, node["visits"])) / max(1, child["visits"])
            )
            ucb_scores[action] = exploitation + exploration
        
        # Select action with highest UCB score
        return max(ucb_scores.items(), key=lambda x: x[1])[0]
    
    def _get_action_probs(self, node: Dict[str, Any]) -> Dict[torch.Tensor, float]:
        """Get action probabilities from visit counts."""
        if not node["children"]:
            return {}
        
        total_visits = sum(child["visits"] for child in node["children"].values())
        probs = {
            action: (child["visits"] / total_visits) ** (1 / self.temperature)
            for action, child in node["children"].items()
        }
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        
        return probs
    
    def _sample_action(
        self,
        action_probs: Dict[torch.Tensor, float],
    ) -> torch.Tensor:
        """Sample action from probability distribution."""
        if not action_probs:
            return self.policy.act()
        
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]
    
    def _get_state_key(self, state: torch.Tensor) -> str:
        """Get unique key for state."""
        return str(state.cpu().numpy().tobytes())
    
    def _goal_reward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute reward based on distance to goal."""
        return -torch.norm(state - goal).item()
    
    def reset(self) -> None:
        """Reset planner state."""
        self.tree = {} 