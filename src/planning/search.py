from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

class TreeSearch:
    """Base class for tree search algorithms."""
    
    def __init__(
        self,
        world_model: Any,
        policy: Any,
        horizon: int,
        num_simulations: int,
    ):
        self.world_model = world_model
        self.policy = policy
        self.horizon = horizon
        self.num_simulations = num_simulations
    
    def search(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Search for best action sequence."""
        raise NotImplementedError
    
    def _rollout(
        self,
        state: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[float, List[torch.Tensor]]:
        """Perform a rollout from current state."""
        total_reward = 0.0
        actions = []
        
        for _ in range(self.horizon):
            # Get action from policy
            action = self.policy.act(state)
            actions.append(action)
            
            # Simulate next state
            next_state, reward = self.world_model.predict(state, action)
            if goal is not None:
                reward += self._goal_reward(next_state, goal)
            
            total_reward += reward
            state = next_state
        
        return total_reward, actions
    
    def _goal_reward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> float:
        """Compute reward based on distance to goal."""
        return -torch.norm(state - goal).item()

class BeamSearch(TreeSearch):
    """Beam search for action sequence planning."""
    
    def __init__(
        self,
        world_model: Any,
        policy: Any,
        horizon: int,
        num_simulations: int,
        beam_size: int,
    ):
        super().__init__(
            world_model=world_model,
            policy=policy,
            horizon=horizon,
            num_simulations=num_simulations,
        )
        self.beam_size = beam_size
    
    def search(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Search using beam search."""
        # Initialize beam
        beam = [(observation, [], 0.0)]  # (state, actions, reward)
        
        for _ in range(self.horizon):
            new_beam = []
            
            # Expand each state in beam
            for state, actions, reward in beam:
                # Get action from policy
                action = self.policy.act(state)
                
                # Simulate next state
                next_state, step_reward = self.world_model.predict(state, action)
                if goal is not None:
                    step_reward += self._goal_reward(next_state, goal)
                
                new_beam.append((
                    next_state,
                    actions + [action],
                    reward + step_reward,
                ))
            
            # Select top-k states
            beam = sorted(new_beam, key=lambda x: x[2], reverse=True)[:self.beam_size]
        
        # Select best action sequence
        best_state, best_actions, best_reward = beam[0]
        
        return best_actions[0], {
            "reward": best_reward,
            "actions": best_actions,
        }

class MPC(TreeSearch):
    """Model Predictive Control with random shooting."""
    
    def __init__(
        self,
        world_model: Any,
        policy: Any,
        horizon: int,
        num_simulations: int,
    ):
        super().__init__(
            world_model=world_model,
            policy=policy,
            horizon=horizon,
            num_simulations=num_simulations,
        )
    
    def search(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Search using random shooting."""
        best_reward = float("-inf")
        best_action = None
        all_rewards = []
        
        for _ in range(self.num_simulations):
            # Get action from policy
            action = self.policy.act(observation)
            
            # Simulate trajectory
            reward, _ = self._rollout(observation, goal)
            all_rewards.append(reward)
            
            # Update best action
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action, {
            "reward": best_reward,
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
        }

class CEM(TreeSearch):
    """Cross-Entropy Method for action sequence optimization."""
    
    def __init__(
        self,
        world_model: Any,
        policy: Any,
        horizon: int,
        num_simulations: int,
        num_elites: int,
        num_iterations: int,
    ):
        super().__init__(
            world_model=world_model,
            policy=policy,
            horizon=horizon,
            num_simulations=num_simulations,
        )
        self.num_elites = num_elites
        self.num_iterations = num_iterations
    
    def search(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Search using CEM."""
        # Initialize action distribution
        action_dim = self.policy.action_dim
        mean = torch.zeros(action_dim)
        std = torch.ones(action_dim)
        
        for _ in range(self.num_iterations):
            # Sample actions
            actions = torch.normal(mean, std, size=(self.num_simulations, action_dim))
            
            # Evaluate trajectories
            rewards = []
            for action in actions:
                # Simulate trajectory
                reward, _ = self._rollout(observation, goal)
                rewards.append(reward)
            
            # Select elites
            elite_indices = np.argsort(rewards)[-self.num_elites:]
            elite_actions = actions[elite_indices]
            
            # Update distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0)
        
        # Return best action
        best_action = mean
        best_reward = max(rewards)
        
        return best_action, {
            "reward": best_reward,
            "mean": mean,
            "std": std,
        } 