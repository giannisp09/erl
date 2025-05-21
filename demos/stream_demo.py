import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from erl.experience.stream import Experience, ExperienceReplay, ContinuousStream

def create_simple_environment():
    """Create a simple environment for demonstration."""
    class SimpleEnv:
        def __init__(self):
            self.state = torch.zeros(4)
            self.goal = torch.ones(4)
            
        def reset(self):
            self.state = torch.zeros(4)
            return self.state
            
        def step(self, action):
            # Simple dynamics: state += action
            self.state = self.state + action
            
            # Reward is negative distance to goal
            reward = -torch.norm(self.state - self.goal).item()
            
            # Episode ends when close to goal
            done = torch.norm(self.state - self.goal) < 0.1
            
            return self.state, reward, done, {}
    
    return SimpleEnv()

def main():
    # Create output directory
    output_dir = Path("outputs/stream_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment
    env = create_simple_environment()
    
    # Initialize experience replay buffer
    replay_buffer = ExperienceReplay(
        capacity=1000,
        alpha=0.6,  # Priority exponent
        beta=0.4,   # Importance sampling exponent
        beta_increment=0.001
    )
    
    # Initialize continuous stream
    stream = ContinuousStream(
        replay_buffer=replay_buffer,
        batch_size=32,
        min_experience=100
    )
    
    # Training parameters
    num_episodes = 100
    max_steps = 50
    episode_rewards = []
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        for step in range(max_steps):
            # Generate random action (in a real scenario, this would come from a policy)
            action = torch.randn(4) * 0.1  # Small random actions
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            
            # Create experience
            experience = Experience(
                observation=state,
                action=action,
                reward=reward,
                next_observation=next_state,
                done=done,
                info={"episode": episode, "step": step}
            )
            
            # Add to replay buffer
            replay_buffer.push(experience)
            
            # Add to continuous stream
            stream.add_experience(
                observation=state,
                action=action,
                reward=reward,
                next_observation=next_state,
                done=done,
                info={"episode": episode, "step": step}
            )
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        # Sample and train if enough experience
        if stream.can_sample():
            # Sample batch from stream
            experiences, indices, weights = stream.sample_batch()
            
            # In a real scenario, you would use these experiences to train your model
            # For this demo, we'll just update priorities based on reward
            new_priorities = np.array([exp.reward for exp in experiences])
            stream.update_priorities(indices, new_priorities)
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.grid(True)
    plt.savefig(output_dir / 'training_progress.png')
    plt.close()
    
    # Print some statistics
    print("\nTraining Statistics:")
    print(f"Total experiences collected: {len(replay_buffer)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best reward: {np.max(episode_rewards):.2f}")
    print(f"Worst reward: {np.min(episode_rewards):.2f}")
    
    # Demonstrate sampling from the replay buffer
    print("\nDemonstrating replay buffer sampling:")
    batch_size = 5
    experiences, indices, weights = replay_buffer.sample(batch_size)
    
    print(f"Sampled {len(experiences)} experiences:")
    for i, exp in enumerate(experiences):
        print(f"Experience {i + 1}:")
        print(f"  State: {exp.observation.numpy()}")
        print(f"  Action: {exp.action.numpy()}")
        print(f"  Reward: {exp.reward:.2f}")
        print(f"  Next state: {exp.next_observation.numpy()}")
        print(f"  Done: {exp.done}")
        print(f"  Weight: {weights[i]:.4f}")
        print()

if __name__ == "__main__":
    main() 