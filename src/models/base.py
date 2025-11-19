from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp


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
