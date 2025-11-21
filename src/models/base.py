from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class ModelConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int = 1
    precision: str = "bfloat16"  # "float32" or "bfloat16"
    param_dtype: str | None = None
    use_layer_norm: bool = False


class BaseSequenceModel:
    """Shared interface for sequence models predicting next token/value.

    All models implement:
    - initialize / parameters
    - apply(params, x, mask) -> logits or predictions for next-step
    - loss/metrics will be composed externally in training code.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def initialize(self, key: jax.Array) -> Any:
        raise NotImplementedError

    def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def parameter_count(self, params: Any) -> int:
        """Total number of scalar parameters."""
        return int(sum(int(jnp.size(p)) for p in jax.tree_util.tree_leaves(params)))

    def receptive_field(self) -> int:
        """Proxy for effective temporal receptive field.

        Default: num_layers for recurrent variants; override as needed.
        """
        return max(1, self.config.num_layers)

    def save_weights(self, params: Any, path: str | Path) -> str:
        """Serialize model parameters to disk.

        Args:
            params: PyTree of model parameters (typically JAX arrays).
            path: Destination file path. Parent directories are created if needed.

        Returns:
            The absolute path to the written checkpoint file.
        """
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        cpu_params = jax.device_get(params)
        numpy_params = jax.tree_util.tree_map(lambda x: np.asarray(x), cpu_params)
        payload = {
            "model_architecture": self.__class__.__name__,
            "model_config": self.config,
            "params": numpy_params,
            "format_version": 1,
        }
        with checkpoint_path.open("wb") as f:
            pickle.dump(payload, f)
        return str(checkpoint_path)

    def load_weights(self, path: str | Path) -> Any:
        """Load serialized parameters previously saved with `save_weights`.

        Args:
            path: Path to a checkpoint file created by `save_weights`.

        Returns:
            A PyTree of parameters restored onto the current device.
        """
        checkpoint_path = Path(path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "params" in payload:
            payload = payload["params"]
        return jax.tree_util.tree_map(lambda x: jnp.asarray(x), payload)
