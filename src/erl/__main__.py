import typer
from pathlib import Path
import yaml
from typing import Optional

app = typer.Typer()

@app.command()
def train(
    config: Path = typer.Option(
        "configs/default.yaml",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        "outputs",
        help="Directory to save outputs",
        file_okay=False,
        dir_okay=True,
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run training on",
    ),
    seed: Optional[int] = typer.Option(
        None,
        help="Random seed for reproducibility",
    ),
):
    """Train an agent using the ERL framework."""
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize components
    from experience.stream import ExperienceStream
    from models.world_model import WorldModel
    from models.policy import Policy
    from rl.ppo import PPO
    from training.trainer import Trainer
    
    stream = ExperienceStream(**cfg["experience"])
    world_model = WorldModel(**cfg["models"]["world_model"])
    policy = Policy(**cfg["models"]["policy"])
    ppo = PPO(policy, **cfg["rl"]["ppo"])
    trainer = Trainer(
        stream=stream,
        world_model=world_model,
        policy=policy,
        ppo=ppo,
        output_dir=output_dir,
        **cfg["training"],
    )
    
    # Train
    trainer.train()

@app.command()
def evaluate(
    checkpoint: Path = typer.Option(
        ...,
        help="Path to checkpoint file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    config: Path = typer.Option(
        "configs/default.yaml",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    num_episodes: int = typer.Option(
        10,
        help="Number of episodes to evaluate",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    ),
):
    """Evaluate a trained agent."""
    # Load configuration
    with open(config) as f:
        cfg = yaml.safe_load(f)
    
    # Initialize components
    from models.policy import Policy
    from training.trainer import Trainer
    
    policy = Policy(**cfg["models"]["policy"])
    trainer = Trainer(
        policy=policy,
        output_dir=Path("outputs"),
        **cfg["training"],
    )
    
    # Load checkpoint
    trainer.load_checkpoint(checkpoint)
    
    # Evaluate
    metrics = trainer.evaluate(num_episodes=num_episodes)
    print(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    app() 