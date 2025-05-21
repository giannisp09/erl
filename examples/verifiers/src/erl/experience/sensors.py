from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import torch
import numpy as np

class Sensor(ABC):
    """Abstract base class for sensors."""
    
    def __init__(self, name: str, observation_space: Dict[str, Any]):
        self.name = name
        self.observation_space = observation_space
        
    @abstractmethod
    def observe(self) -> torch.Tensor:
        """Observe the current state of the environment."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the sensor to its initial state."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the sensor and release resources."""
        pass

class TextSensor(Sensor):
    """Sensor for text observations."""
    
    def __init__(
        self,
        name: str = "text",
        max_length: int = 512,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(
            name=name,
            observation_space={
                "type": "text",
                "max_length": max_length,
                "shape": (max_length,),
            },
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def observe(self) -> torch.Tensor:
        """Tokenize and encode text observation."""
        text = self._get_text()
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(0)
        else:
            # Simple character-level encoding
            chars = list(text[:self.max_length])
            encoding = torch.zeros(self.max_length, dtype=torch.long)
            for i, char in enumerate(chars):
                encoding[i] = ord(char)
            return encoding
            
    @abstractmethod
    def _get_text(self) -> str:
        """Get raw text observation from environment."""
        pass

class ImageSensor(Sensor):
    """Sensor for image observations."""
    
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
            observation_space={
                "type": "image",
                "shape": (channels, height, width),
                "dtype": "float32",
            },
        )
        self.height = height
        self.width = width
        self.channels = channels
        self.transforms = transforms
        
    def observe(self) -> torch.Tensor:
        """Get and preprocess image observation."""
        image = self._get_image()
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        # Ensure correct shape and type
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
            
        if image.dim() == 2:  # Grayscale
            image = image.unsqueeze(0)
        elif image.dim() == 3 and image.shape[0] != self.channels:  # HWC
            image = image.permute(2, 0, 1)
            
        return image.float()
        
    @abstractmethod
    def _get_image(self) -> np.ndarray:
        """Get raw image observation from environment."""
        pass

class APISensor(Sensor):
    """Sensor for API-based observations."""
    
    def __init__(
        self,
        name: str = "api",
        observation_space: Dict[str, Any] = None,
        client: Optional[Any] = None,
    ):
        super().__init__(name=name, observation_space=observation_space)
        self.client = client
        
    def observe(self) -> torch.Tensor:
        """Get observation from API."""
        data = self._get_api_data()
        return self._process_data(data)
        
    @abstractmethod
    def _get_api_data(self) -> Dict[str, Any]:
        """Get raw data from API."""
        pass
        
    @abstractmethod
    def _process_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """Process API data into observation tensor."""
        pass 