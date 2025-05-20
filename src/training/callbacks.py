from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import torch

class Callback(ABC):
    """Base class for training callbacks."""
    
    @abstractmethod
    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each training step."""
        pass
    
    @abstractmethod
    def on_episode_end(
        self,
        step: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        """Called at the end of each episode."""
        pass

class CheckpointCallback(Callback):
    """Callback for saving model checkpoints."""
    
    def __init__(
        self,
        output_dir: Path,
        save_interval: int,
        keep_last_n: int = 3,
    ):
        self.output_dir = output_dir
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.last_checkpoints = []
    
    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        if step % self.save_interval == 0:
            self._save_checkpoint(step, metrics)
    
    def _save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        checkpoint = {
            "step": step,
            "metrics": metrics,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        
        path = self.output_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, path)
        
        # Update last checkpoints list
        self.last_checkpoints.append(path)
        if len(self.last_checkpoints) > self.keep_last_n:
            old_checkpoint = self.last_checkpoints.pop(0)
            old_checkpoint.unlink()

class EarlyStoppingCallback(Callback):
    """Callback for early stopping based on validation metrics."""
    
    def __init__(
        self,
        patience: int,
        min_delta: float = 0.0,
        metric: str = "eval/reward",
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
    
    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        if self.metric not in metrics:
            return
        
        current_metric = metrics[self.metric]
        
        if self.mode == "max":
            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        else:
            if current_metric < self.best_metric - self.min_delta:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True

class LearningRateSchedulerCallback(Callback):
    """Callback for learning rate scheduling."""
    
    def __init__(
        self,
        scheduler: Any,
        metric: Optional[str] = None,
    ):
        self.scheduler = scheduler
        self.metric = metric
    
    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        if self.metric is not None:
            if self.metric in metrics:
                self.scheduler.step(metrics[self.metric])
        else:
            self.scheduler.step()

class ProgressBarCallback(Callback):
    """Callback for displaying training progress."""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_length = 0
    
    def on_step_end(
        self,
        step: int,
        metrics: Dict[str, float],
    ) -> None:
        self.current_step = step
        self.episode_reward += metrics.get("reward", 0.0)
        self.episode_length += 1
    
    def on_episode_end(
        self,
        step: int,
        episode_reward: float,
        episode_length: int,
    ) -> None:
        print(
            f"Step: {step}/{self.total_steps} | "
            f"Episode Reward: {episode_reward:.2f} | "
            f"Episode Length: {episode_length}"
        )
        self.episode_reward = 0.0
        self.episode_length = 0 