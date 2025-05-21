import logging
import wandb
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

class Logger:
    """Unified logging interface for different backends."""
    
    def __init__(
        self,
        output_dir: Path,
        wandb_config: Optional[Dict[str, Any]] = None,
        tensorboard: bool = True,
        console: bool = True,
    ):
        self.output_dir = output_dir
        self.wandb_config = wandb_config
        self.tensorboard = tensorboard
        self.console = console
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize console logger
        if console:
            self.console_logger = logging.getLogger("erl")
            self.console_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.console_logger.addHandler(handler)
        
        # Initialize wandb
        if wandb_config is not None:
            wandb.init(**wandb_config)
        
        # Initialize tensorboard
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all enabled backends."""
        if self.console:
            for k, v in metrics.items():
                self.console_logger.info(f"{prefix}{k}: {v}")
        
        if self.wandb_config is not None:
            wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
        
        if self.tensorboard:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}{k}", v, step)
    
    def log_model(
        self,
        model: torch.nn.Module,
        name: str,
        step: int,
    ) -> None:
        """Log model weights to wandb."""
        if self.wandb_config is not None:
            torch.save(
                model.state_dict(),
                self.output_dir / f"{name}_{step}.pt",
            )
            wandb.save(str(self.output_dir / f"{name}_{step}.pt"))
    
    def log_histogram(
        self,
        values: np.ndarray,
        name: str,
        step: int,
    ) -> None:
        """Log histogram to tensorboard."""
        if self.tensorboard:
            self.tb_writer.add_histogram(name, values, step)
    
    def close(self) -> None:
        """Close all logging backends."""
        if self.wandb_config is not None:
            wandb.finish()
        if self.tensorboard:
            self.tb_writer.close() 