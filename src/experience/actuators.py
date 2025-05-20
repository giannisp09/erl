from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np

class Actuator(ABC):
    """Abstract base class for actuators."""
    
    def __init__(self, name: str, action_space: Dict[str, Any]):
        self.name = name
        self.action_space = action_space
        
    @abstractmethod
    def act(self, action: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action tensor to execute
            
        Returns:
            Tuple of (success, info) where info contains additional metrics
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the actuator to its initial state."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the actuator and release resources."""
        pass

class TextActuator(Actuator):
    """Actuator for text-based actions."""
    
    def __init__(
        self,
        name: str = "text",
        max_length: int = 512,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            action_space={
                "type": "text",
                "max_length": max_length,
                "shape": (max_length,),
            },
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def act(self, action: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """Execute text action."""
        if self.tokenizer is not None:
            text = self.tokenizer.decode(action.tolist(), skip_special_tokens=True)
        else:
            # Simple character-level decoding
            chars = [chr(int(x)) for x in action.tolist() if x > 0]
            text = "".join(chars)
            
        return self._execute_text(text)
        
    @abstractmethod
    def _execute_text(self, text: str) -> Tuple[bool, Dict[str, Any]]:
        """Execute text action in environment."""
        pass

class ImageActuator(Actuator):
    """Actuator for image-based actions."""
    
    def __init__(
        self,
        name: str = "image",
        height: int = 224,
        width: int = 224,
        channels: int = 3,
        transforms: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            action_space={
                "type": "image",
                "shape": (channels, height, width),
                "dtype": "float32",
            },
        )
        self.height = height
        self.width = width
        self.channels = channels
        self.transforms = transforms
        
    def act(self, action: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """Execute image action."""
        # Ensure correct shape and type
        if action.dim() == 2:  # Grayscale
            action = action.unsqueeze(0)
        elif action.dim() == 3 and action.shape[0] != self.channels:  # HWC
            action = action.permute(2, 0, 1)
            
        # Convert to numpy for display/execution
        image = action.detach().cpu().numpy()
        if self.channels == 1:
            image = image.squeeze(0)
        else:
            image = image.transpose(1, 2, 0)
            
        return self._execute_image(image)
        
    @abstractmethod
    def _execute_image(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Execute image action in environment."""
        pass

class APIActuator(Actuator):
    """Actuator for API-based actions."""
    
    def __init__(
        self,
        name: str = "api",
        action_space: Dict[str, Any] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(name=name, action_space=action_space)
        self.client = client
        
    def act(self, action: torch.Tensor) -> Tuple[bool, Dict[str, Any]]:
        """Execute API action."""
        data = self._process_action(action)
        return self._execute_api(data)
        
    @abstractmethod
    def _process_action(self, action: torch.Tensor) -> Dict[str, Any]:
        """Process action tensor into API request data."""
        pass
        
    @abstractmethod
    def _execute_api(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute API action."""
        pass 