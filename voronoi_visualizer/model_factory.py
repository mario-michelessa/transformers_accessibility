"""Factory for creating LLM models."""

from .base_llm import BaseLLM
from .hf_llm import AutoCausalLLM


def create_model(model_name: str, model_type: str = "auto") -> BaseLLM:
    if model_type.lower() in {"auto", "hf", "auto_hf", "generic"}:
        return AutoCausalLLM(model_name)
    raise ValueError(f"Unsupported model type: {model_type}")

def load_model(model_name: str, model_type: str = "auto") -> BaseLLM:
    model = create_model(model_name, model_type)
    model.load_model()
    return model
