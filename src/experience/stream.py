from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class Experience:
    """A single experience tuple in the stream."""
    observation: torch.Tensor
    action: torch.Tensor
    reward: float
    next_observation: torch.Tensor
    done: bool
    info: Dict[str, Any]

class ExperienceReplay(Dataset):
    """Prioritized experience replay buffer for continuous learning."""
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory: List[Experience] = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        
    def push(self, experience: Experience, priority: Optional[float] = None) -> None:
        """Add a new experience to the buffer."""
        if priority is None:
            priority = max(self.priorities) if len(self.memory) > 0 else 1.0
            
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample a batch of experiences with importance sampling weights."""
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from empty buffer")
            
        probs = self.priorities[:len(self.memory)] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        experiences = [self.memory[idx] for idx in indices]
        
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self) -> int:
        return len(self.memory)

class ContinuousStream:
    """Manages continuous experience streams for learning."""
    
    def __init__(
        self,
        replay_buffer: ExperienceReplay,
        batch_size: int,
        min_experience: int,
    ):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.min_experience = min_experience
        self.step_count = 0
        
    def add_experience(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_observation: torch.Tensor,
        done: bool,
        info: Dict[str, Any],
        priority: Optional[float] = None,
    ) -> None:
        """Add a new experience to the stream."""
        experience = Experience(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            info=info,
        )
        self.replay_buffer.push(experience, priority)
        self.step_count += 1
        
    def can_sample(self) -> bool:
        """Check if we have enough experience to start sampling."""
        return len(self.replay_buffer) >= self.min_experience
        
    def sample_batch(self) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample a batch of experiences for training."""
        if not self.can_sample():
            raise ValueError("Not enough experience to sample")
        return self.replay_buffer.sample(self.batch_size)
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        self.replay_buffer.update_priorities(indices, priorities) 