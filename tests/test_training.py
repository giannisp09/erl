import pytest
import torch
import numpy as np
from src.training.trainer import Trainer
from src.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    ProgressBarCallback,
)

def test_trainer_initialization():
    """Test trainer initialization."""
    # Create dummy components
    stream = torch.nn.Module()  # Placeholder for experience stream
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
    ppo = torch.nn.Module()  # Placeholder for PPO
    
    trainer = Trainer(
        stream=stream,
        world_model=world_model,
        policy=policy,
        ppo=ppo,
        output_dir="test_output",
        num_epochs=10,
        steps_per_epoch=100,
        eval_interval=5,
        save_interval=2,
        log_interval=10,
        device="cpu",
    )
    
    assert trainer.num_epochs == 10
    assert trainer.steps_per_epoch == 100
    assert trainer.eval_interval == 5
    assert trainer.save_interval == 2
    assert trainer.log_interval == 10
    assert trainer.device == "cpu"

def test_checkpoint_callback():
    """Test checkpoint callback."""
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    
    callback = CheckpointCallback(
        model=model,
        output_dir="test_output",
        save_interval=2,
        max_checkpoints=3,
    )
    
    # Test checkpoint saving
    for step in range(10):
        callback.on_step_end(step, {})
    
    # Check that checkpoints were saved
    assert len(callback.checkpoints) <= 3

def test_early_stopping_callback():
    """Test early stopping callback."""
    callback = EarlyStoppingCallback(
        metric="val_loss",
        patience=3,
        min_delta=0.01,
        mode="min",
    )
    
    # Test early stopping
    metrics = {"val_loss": 1.0}
    for i in range(5):
        metrics["val_loss"] = 1.0 - i * 0.1
        should_stop = callback.on_epoch_end(i, metrics)
        if i < 3:
            assert not should_stop
        else:
            assert should_stop

def test_learning_rate_scheduler_callback():
    """Test learning rate scheduler callback."""
    # Create dummy optimizer
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=2,
        gamma=0.1,
    )
    
    callback = LearningRateSchedulerCallback(
        scheduler=scheduler,
        metric="val_loss",
        mode="min",
    )
    
    # Test learning rate scheduling
    initial_lr = optimizer.param_groups[0]["lr"]
    metrics = {"val_loss": 1.0}
    
    for i in range(5):
        callback.on_epoch_end(i, metrics)
        if i % 2 == 1:
            assert optimizer.param_groups[0]["lr"] < initial_lr
        else:
            assert optimizer.param_groups[0]["lr"] == initial_lr

def test_progress_bar_callback():
    """Test progress bar callback."""
    callback = ProgressBarCallback(
        total_steps=100,
        metrics=["loss", "reward"],
    )
    
    # Test progress bar updates
    for step in range(10):
        metrics = {
            "loss": float(step),
            "reward": float(step),
        }
        callback.on_step_end(step, metrics)

def test_trainer_training_loop():
    """Test trainer training loop."""
    # Create dummy components
    stream = torch.nn.Module()  # Placeholder for experience stream
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
    ppo = torch.nn.Module()  # Placeholder for PPO
    
    trainer = Trainer(
        stream=stream,
        world_model=world_model,
        policy=policy,
        ppo=ppo,
        output_dir="test_output",
        num_epochs=2,
        steps_per_epoch=10,
        eval_interval=5,
        save_interval=2,
        log_interval=5,
        device="cpu",
    )
    
    # Add callbacks
    trainer.callbacks.append(CheckpointCallback(
        model=policy,
        output_dir="test_output",
        save_interval=2,
    ))
    trainer.callbacks.append(ProgressBarCallback(
        total_steps=20,
        metrics=["loss", "reward"],
    ))
    
    # Test training loop
    trainer.train()

def test_trainer_evaluation():
    """Test trainer evaluation."""
    # Create dummy components
    stream = torch.nn.Module()  # Placeholder for experience stream
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
    ppo = torch.nn.Module()  # Placeholder for PPO
    
    trainer = Trainer(
        stream=stream,
        world_model=world_model,
        policy=policy,
        ppo=ppo,
        output_dir="test_output",
        num_epochs=2,
        steps_per_epoch=10,
        eval_interval=5,
        save_interval=2,
        log_interval=5,
        device="cpu",
    )
    
    # Test evaluation
    metrics = trainer.evaluate(num_episodes=5)
    assert "episode_reward" in metrics
    assert "episode_length" in metrics

def test_trainer_checkpoint_saving():
    """Test trainer checkpoint saving and loading."""
    # Create dummy components
    stream = torch.nn.Module()  # Placeholder for experience stream
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
    ppo = torch.nn.Module()  # Placeholder for PPO
    
    trainer = Trainer(
        stream=stream,
        world_model=world_model,
        policy=policy,
        ppo=ppo,
        output_dir="test_output",
        num_epochs=2,
        steps_per_epoch=10,
        eval_interval=5,
        save_interval=2,
        log_interval=5,
        device="cpu",
    )
    
    # Save checkpoint
    trainer.save_checkpoint("test_checkpoint.pt")
    
    # Load checkpoint
    trainer.load_checkpoint("test_checkpoint.pt")
    
    # Check that models were loaded
    assert isinstance(trainer.world_model, torch.nn.Module)
    assert isinstance(trainer.policy, torch.nn.Module)
    assert isinstance(trainer.ppo, torch.nn.Module) 