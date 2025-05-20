import pytest
import torch
import numpy as np
from src.experience.stream import Experience, ExperienceReplay, ContinuousStream

def test_experience_creation():
    """Test creation of Experience objects."""
    # Create sample tensors
    obs = torch.randn(4)
    action = torch.randn(2)
    next_obs = torch.randn(4)
    info = {"test": 1}
    
    # Create experience
    exp = Experience(
        observation=obs,
        action=action,
        reward=1.0,
        next_observation=next_obs,
        done=False,
        info=info
    )
    
    # Test attributes
    assert torch.equal(exp.observation, obs)
    assert torch.equal(exp.action, action)
    assert exp.reward == 1.0
    assert torch.equal(exp.next_observation, next_obs)
    assert exp.done is False
    assert exp.info == info

def test_experience_replay_initialization():
    """Test initialization of ExperienceReplay."""
    capacity = 1000
    alpha = 0.6
    beta = 0.4
    beta_increment = 0.001
    
    buffer = ExperienceReplay(
        capacity=capacity,
        alpha=alpha,
        beta=beta,
        beta_increment=beta_increment
    )
    
    assert buffer.capacity == capacity
    assert buffer.alpha == alpha
    assert buffer.beta == beta
    assert buffer.beta_increment == beta_increment
    assert len(buffer.memory) == 0
    assert len(buffer.priorities) == capacity
    assert buffer.position == 0

def test_experience_replay_push():
    """Test pushing experiences to the replay buffer."""
    buffer = ExperienceReplay(capacity=3)
    
    # Create test experiences
    exp1 = Experience(
        observation=torch.tensor([1.0]),
        action=torch.tensor([0.0]),
        reward=1.0,
        next_observation=torch.tensor([2.0]),
        done=False,
        info={}
    )
    
    exp2 = Experience(
        observation=torch.tensor([2.0]),
        action=torch.tensor([1.0]),
        reward=2.0,
        next_observation=torch.tensor([3.0]),
        done=False,
        info={}
    )
    
    # Push experiences
    buffer.push(exp1)
    buffer.push(exp2)
    
    assert len(buffer) == 2
    assert buffer.position == 2
    
    # Test priority assignment
    assert buffer.priorities[0] > 0
    assert buffer.priorities[1] > 0

def test_experience_replay_sample():
    """Test sampling from the replay buffer."""
    buffer = ExperienceReplay(capacity=10)
    
    # Fill buffer with test experiences
    for i in range(5):
        exp = Experience(
            observation=torch.tensor([float(i)]),
            action=torch.tensor([float(i)]),
            reward=float(i),
            next_observation=torch.tensor([float(i+1)]),
            done=(i == 4),
            info={"index": i}
        )
        buffer.push(exp)
    
    # Sample batch
    batch_size = 3
    experiences, indices, weights = buffer.sample(batch_size)
    
    assert len(experiences) == batch_size
    assert len(indices) == batch_size
    assert len(weights) == batch_size
    assert all(0 <= idx < len(buffer) for idx in indices)
    assert all(w > 0 for w in weights)

def test_experience_replay_update_priorities():
    """Test updating priorities in the replay buffer."""
    buffer = ExperienceReplay(capacity=10)
    
    # Add experiences
    for i in range(5):
        exp = Experience(
            observation=torch.tensor([float(i)]),
            action=torch.tensor([float(i)]),
            reward=float(i),
            next_observation=torch.tensor([float(i+1)]),
            done=False,
            info={}
        )
        buffer.push(exp)
    
    # Sample and update priorities
    experiences, indices, _ = buffer.sample(3)
    new_priorities = np.array([0.5, 0.7, 0.9])
    buffer.update_priorities(indices, new_priorities)
    
    # Verify priorities were updated
    for idx, priority in zip(indices, new_priorities):
        assert buffer.priorities[idx] == priority

def test_continuous_stream_initialization():
    """Test initialization of ContinuousStream."""
    replay_buffer = ExperienceReplay(capacity=1000)
    batch_size = 32
    min_experience = 100
    
    stream = ContinuousStream(
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        min_experience=min_experience
    )
    
    assert stream.replay_buffer == replay_buffer
    assert stream.batch_size == batch_size
    assert stream.min_experience == min_experience
    assert stream.step_count == 0

def test_continuous_stream_add_experience():
    """Test adding experiences to the continuous stream."""
    stream = ContinuousStream(
        replay_buffer=ExperienceReplay(capacity=1000),
        batch_size=32,
        min_experience=100
    )
    
    # Add experience
    obs = torch.randn(4)
    action = torch.randn(2)
    next_obs = torch.randn(4)
    
    stream.add_experience(
        observation=obs,
        action=action,
        reward=1.0,
        next_observation=next_obs,
        done=False,
        info={}
    )
    
    assert stream.step_count == 1
    assert len(stream.replay_buffer) == 1

def test_continuous_stream_can_sample():
    """Test the can_sample method of ContinuousStream."""
    stream = ContinuousStream(
        replay_buffer=ExperienceReplay(capacity=1000),
        batch_size=32,
        min_experience=100
    )
    
    # Initially should not be able to sample
    assert not stream.can_sample()
    
    # Add enough experiences
    for _ in range(100):
        stream.add_experience(
            observation=torch.randn(4),
            action=torch.randn(2),
            reward=1.0,
            next_observation=torch.randn(4),
            done=False,
            info={}
        )
    
    assert stream.can_sample()

def test_continuous_stream_sample_batch():
    """Test sampling batches from the continuous stream."""
    stream = ContinuousStream(
        replay_buffer=ExperienceReplay(capacity=1000),
        batch_size=32,
        min_experience=100
    )
    
    # Fill buffer with experiences
    for i in range(150):
        stream.add_experience(
            observation=torch.tensor([float(i)]),
            action=torch.tensor([float(i)]),
            reward=float(i),
            next_observation=torch.tensor([float(i+1)]),
            done=(i == 149),
            info={"index": i}
        )
    
    # Sample batch
    experiences, indices, weights = stream.sample_batch()
    
    assert len(experiences) == stream.batch_size
    assert len(indices) == stream.batch_size
    assert len(weights) == stream.batch_size

def test_continuous_stream_update_priorities():
    """Test updating priorities in the continuous stream."""
    stream = ContinuousStream(
        replay_buffer=ExperienceReplay(capacity=1000),
        batch_size=32,
        min_experience=100
    )
    
    # Add experiences
    for i in range(150):
        stream.add_experience(
            observation=torch.tensor([float(i)]),
            action=torch.tensor([float(i)]),
            reward=float(i),
            next_observation=torch.tensor([float(i+1)]),
            done=False,
            info={}
        )
    
    # Sample and update priorities
    experiences, indices, _ = stream.sample_batch()
    new_priorities = np.random.rand(stream.batch_size)
    stream.update_priorities(indices, new_priorities)
    
    # Verify priorities were updated
    for idx, priority in zip(indices, new_priorities):
        assert stream.replay_buffer.priorities[idx] == priority 

if __name__ == "__main__":
    test_experience_creation()
    test_experience_replay_initialization()
    test_experience_replay_push()
    test_experience_replay_sample()
    test_experience_replay_update_priorities()
    #test_continuous_stream_initialization()