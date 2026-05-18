from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np


class BaseLLM(ABC):
    """Base class for LLM embedding extraction."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self) -> None:
        pass
    
    @abstractmethod
    def extract_embeddings(self, 
                          text: str, 
                          layer_indices: Optional[List[int]] = None,
                          last_token: Optional[bool] = True) -> Dict[str, torch.Tensor]:
        pass
    
    @abstractmethod
    def extract_last_token_embedding(self, 
                                   text: str, 
                                   before_decoder: bool = True) -> torch.Tensor:
        pass
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self.tokenizer(text, return_tensors="pt")

    @abstractmethod
    def get_vocabulary(self) -> Dict[int, str]:
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        pass
    
    @abstractmethod
    def get_layer_names(self) -> List[str]:
        pass

