from typing import Any, Dict, List, Optional
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_llm import BaseLLM


MODEL_ROOT = Path(os.environ.get("LLM_VIS_MODEL_ROOT", "models/llms-theory")).expanduser()
LOCAL_MODEL_ROOTS = [MODEL_ROOT]
MODEL_ROOT_MARKERS = ["models/llms-theory", "llms-theory"]


class AutoCausalLLM(BaseLLM):
    """
    Generic Hugging Face causal LM loader that works for locally downloaded models
    (e.g., Gemma, LLaMA, Pythia) saved via download_llms.py.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.resolved_path: Optional[str] = None

    def load_model(self) -> None:
        load_id, local_only = _resolve_local_model_path(self.model_name)
        self.resolved_path = load_id if local_only else None

        print(f"Loading {load_id} with AutoModelForCausalLM... (local_only={local_only})")
        common_kwargs = {"trust_remote_code": True, "local_files_only": local_only}
        self.tokenizer = AutoTokenizer.from_pretrained(load_id, **common_kwargs)
        if self.tokenizer.pad_token is None:
            # Prefer eos/bos/unk as padding if pad is missing
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.bos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.bos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            load_id,
            output_hidden_states=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            **common_kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def extract_embeddings(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None,
        last_token: Optional[bool] = True,
    ) -> Dict[str, torch.Tensor]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))

        embeddings: Dict[str, torch.Tensor] = {}
        for layer_idx in layer_indices:
            if 0 <= layer_idx < len(hidden_states):
                if last_token:
                    embeddings[f"layer_{layer_idx}"] = hidden_states[layer_idx][:, -1, :].cpu().float()
                else:
                    embeddings[f"layer_{layer_idx}"] = hidden_states[layer_idx].cpu().float()
        return embeddings

    def extract_last_token_embedding(self, text: str, before_decoder: bool = True) -> torch.Tensor:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.hidden_states[-1]
        batch_size = last_hidden_states.size(0)
        last_token_embeddings = []
        for i in range(batch_size):
            last_pos = seq_lengths[i].item()
            last_token_embeddings.append(last_hidden_states[i, last_pos, :])

        return torch.stack(last_token_embeddings).cpu()

    def get_layer_names(self) -> List[str]:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return [name for name, _ in self.model.named_modules() if name]

    def get_num_layers(self) -> int:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        cfg = getattr(self.model, "config", None)
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if cfg is not None and hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if val is not None:
                    return int(val)
        # Common module layouts
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        return 0

    def get_output_projection_matrix(self) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        head = None
        if hasattr(self.model, "lm_head"):
            head = self.model.lm_head
        elif hasattr(self.model, "embed_out"):
            head = self.model.embed_out
        elif hasattr(self.model, "get_output_embeddings"):
            head = self.model.get_output_embeddings()

        if head is None:
            raise ValueError("Could not locate output head on model.")

        if isinstance(head, torch.nn.Module) and hasattr(head, "weight"):
            return head.weight.detach().cpu()
        if isinstance(head, torch.Tensor):
            return head.detach().cpu()
        raise ValueError("Output head is not a tensor or has no weight parameter.")

    def get_vocabulary(self) -> Dict[int, str]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        vocab: Dict[int, str] = {}
        for tid in range(len(self.tokenizer)):
            try:
                tok = self.tokenizer.convert_ids_to_tokens(tid)
                tok = tok if tok is not None else self.tokenizer.decode([tid])
            except Exception:
                tok = self.tokenizer.decode([tid]) if hasattr(self.tokenizer, "decode") else f"<unk_{tid}>"
            vocab[tid] = tok
        return vocab


def _suffix_after_markers(path: str) -> Optional[str]:
    for marker in MODEL_ROOT_MARKERS:
        if marker in path:
            suffix = path.split(marker, 1)[1].lstrip(os.sep)
            return suffix
    return None


def _resolve_local_model_path(model_name: str) -> (str, bool):
    """
    Try to resolve to a local directory first (so Hugging Face doesn't treat it as a repo id).
    Returns (path_or_id, local_only_flag).
    """
    if os.path.isdir(model_name):
        return model_name, True

    candidates = []
    suffix = _suffix_after_markers(model_name)
    if suffix:
        for root in LOCAL_MODEL_ROOTS:
            candidates.append(root / suffix)

    # Also try direct join for repo-like names (e.g., EleutherAI/pythia-160m)
    if not os.path.isabs(model_name):
        for root in LOCAL_MODEL_ROOTS:
            candidates.append(root / model_name)

    for cand in candidates:
        if cand.is_dir():
            return str(cand), True

    # Fallback: use the original string; may require network
    return model_name, False
