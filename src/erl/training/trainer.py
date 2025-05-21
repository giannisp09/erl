from typing import Any, Dict, Optional, Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm

from ..experience.stream import ExperienceStream
from ..models.world_model import WorldModel
from ..models.policy import Policy
from ..rl.ppo import PPO
from ..utils.logger import Logger
from ..utils.metrics import MetricsTracker

@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    episode_reward: float
    episode_length: int
    policy_loss: float
    value_loss: float
    curiosity_loss: float
    kl_divergence: float
    forward_loss: float
    inverse_loss: float

class Callback:
    """Base class for training callbacks."""
    
    def on_step_begin(self, step: int) -> None:
        """Called at the beginning of each training step."""
        pass
        
    def on_step_end(
        self,
        step: int,
        metrics: TrainingMetrics,
    ) -> None:
        """Called at the end of each training step."""
        pass
        
    def on_episode_end(
        self,
        step: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        """Called at the end of each episode."""
        pass

class TensorBoardCallback(Callback):
    """Callback for TensorBoard logging."""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        
    def on_step_end(
        self,
        step: int,
        metrics: TrainingMetrics,
    ) -> None:
        self.writer.add_scalar("loss/policy", metrics.policy_loss, step)
        self.writer.add_scalar("loss/value", metrics.value_loss, step)
        self.writer.add_scalar("loss/curiosity", metrics.curiosity_loss, step)
        self.writer.add_scalar("metrics/kl_divergence", metrics.kl_divergence, step)
        self.writer.add_scalar("loss/forward", metrics.forward_loss, step)
        self.writer.add_scalar("loss/inverse", metrics.inverse_loss, step)
        
    def on_episode_end(
        self,
        step: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        self.writer.add_scalar("episode/reward", episode_reward, step)
        self.writer.add_scalar("episode/length", episode_length, step)

class WandbCallback(Callback):
    """Callback for Weights & Biases logging."""
    
    def __init__(self, project: str, entity: str, config: Dict[str, Any]):
        wandb.init(project=project, entity=entity, config=config)
        
    def on_step_end(
        self,
        step: int,
        metrics: TrainingMetrics,
    ) -> None:
        wandb.log({
            "loss/policy": metrics.policy_loss,
            "loss/value": metrics.value_loss,
            "loss/curiosity": metrics.curiosity_loss,
            "metrics/kl_divergence": metrics.kl_divergence,
            "loss/forward": metrics.forward_loss,
            "loss/inverse": metrics.inverse_loss,
        }, step=step)
        
    def on_episode_end(
        self,
        step: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        wandb.log({
            "episode/reward": episode_reward,
            "episode/length": episode_length,
        }, step=step)

class CheckpointCallback(Callback):
    """Callback for model checkpointing."""
    
    def __init__(
        self,
        save_dir: str,
        save_interval: int = 10000,
        max_checkpoints: int = 5,
    ):
        self.save_dir = Path(save_dir)
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_step_end(
        self,
        step: int,
        metrics: TrainingMetrics,
    ) -> None:
        if step % self.save_interval == 0:
            self._save_checkpoint(step, metrics)
            
    def _save_checkpoint(
        self,
        step: int,
        metrics: TrainingMetrics,
    ) -> None:
        checkpoint = {
            "step": step,
            "metrics": metrics.__dict__,
        }
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata
        metadata_path = self.save_dir / f"checkpoint_{step}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(checkpoint, f)
            
        # Remove old checkpoints
        checkpoints = sorted(
            self.save_dir.glob("checkpoint_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        if len(checkpoints) > self.max_checkpoints:
            for old_checkpoint in checkpoints[:-self.max_checkpoints]:
                old_checkpoint.unlink()
                metadata_path = self.save_dir / f"{old_checkpoint.stem}_metadata.json"
                if metadata_path.exists():
                    metadata_path.unlink()

class Trainer:
    """Trainer for the ERL framework."""
    
    def __init__(
        self,
        stream: ExperienceStream,
        world_model: WorldModel,
        policy: Policy,
        ppo: PPO,
        output_dir: Path,
        num_epochs: int,
        steps_per_epoch: int,
        eval_interval: int,
        save_interval: int,
        log_interval: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.stream = stream
        self.world_model = world_model
        self.policy = policy
        self.ppo = ppo
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.device = device
        
        # Move models to device
        self.world_model.to(device)
        self.policy.to(device)
        
        # Initialize logger
        self.logger = Logger(
            output_dir=output_dir,
            wandb_config={
                "project": "erl",
                "config": {
                    "num_epochs": num_epochs,
                    "steps_per_epoch": steps_per_epoch,
                },
            },
        )
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()
    
    def train(self) -> None:
        """Run the training loop."""
        for epoch in range(self.num_epochs):
            # Training phase
            self._train_epoch(epoch)
            
            # Evaluation phase
            if (epoch + 1) % self.eval_interval == 0:
                self.evaluate()
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch)
    
    def _train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        self.world_model.train()
        self.policy.train()
        
        pbar = tqdm(range(self.steps_per_epoch), desc=f"Epoch {epoch + 1}")
        for step in pbar:
            # Sample batch from experience stream
            batch = self.stream.sample()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Update world model
            world_model_loss = self.world_model.update(batch)
            
            # Update policy using PPO
            ppo_loss = self.ppo.update(
                observations=batch["observations"],
                actions=batch["actions"],
                rewards=batch["rewards"],
                next_observations=batch["next_observations"],
                dones=batch["dones"],
            )
            
            # Update metrics
            metrics = {
                "world_model_loss": world_model_loss,
                "ppo_loss": ppo_loss,
            }
            self.metrics.update(metrics)
            
            # Log metrics
            if (step + 1) % self.log_interval == 0:
                self.logger.log_metrics(
                    self.metrics.get_metrics(window=self.log_interval),
                    step=epoch * self.steps_per_epoch + step,
                    prefix="train/",
                )
                pbar.set_postfix(metrics)
    
    @torch.no_grad()
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy."""
        self.world_model.eval()
        self.policy.eval()
        
        self.metrics.reset()
        
        for _ in range(num_episodes):
            observation = self.stream.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get action from policy
                action = self.policy.act(observation)
                
                # Step environment
                next_observation, reward, done, info = self.stream.step(action)
                
                # Update metrics
                self.metrics.update(
                    {"reward": reward},
                    is_episode_end=done,
                )
                
                observation = next_observation
                episode_reward += reward
        
        # Log evaluation metrics
        metrics = self.metrics.get_episode_metrics()
        self.logger.log_metrics(
            metrics,
            step=self.current_step,
            prefix="eval/",
        )
        
        return metrics
    
    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "world_model": self.world_model.state_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.ppo.optimizer.state_dict(),
        }
        
        torch.save(
            checkpoint,
            self.output_dir / f"checkpoint_{epoch}.pt",
        )
    
    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.policy.load_state_dict(checkpoint["policy"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer"])
    
    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self.epoch * self.steps_per_epoch + self.step
    
    def close(self) -> None:
        """Close resources."""
        self.logger.close()
        self.stream.close() 