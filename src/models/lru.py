from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .base import BaseSequenceModel, ModelConfig
from ..utils.jacobian_features import JacobianFeatureSummary, compute_jacobian_features


@dataclass
class LRURuntimeTensors:
	pre_activations: jnp.ndarray  # [B, T, H]
	hidden_states: jnp.ndarray  # [B, T, H]
	nonlinearity_jacobian_diag: jnp.ndarray  # [B, T, H]


def _resolve_dtype(name: str | None) -> jnp.dtype:
	if name is None:
		return jnp.float32
	name = name.lower()
	if name in ("float32", "fp32"):
		return jnp.float32
	if name in ("bfloat16", "bf16"):
		return jnp.bfloat16
	if name in ("float16", "fp16"):
		return jnp.float16
	raise ValueError(f"Unsupported dtype '{name}'.")


class LinearRecurrentUnit(BaseSequenceModel):
	"""Minimal LRU-like linear recurrent model.

	x_{t+1} = A x_t + B u_t
	y_t = C x_t
	Where u_t is the input, y_t predicts next value/token.
	"""

	def __init__(self, config: ModelConfig, mlp_hidden_dim: int | None = None):
		super().__init__(config)
		self.state_dim = config.hidden_dim
		self.param_dtype = _resolve_dtype(config.param_dtype or config.precision)
		self.mlp_hidden_dim = mlp_hidden_dim or config.hidden_dim

	def initialize(self, key: jax.Array) -> Any:
		kA, kB, kM1, kM2 = jax.random.split(key, 4)
		A = jax.random.normal(kA, (self.state_dim, self.state_dim), dtype=jnp.float32) * (
			1.0 / jnp.sqrt(self.state_dim)
		)
		# Stabilize A via spectral norm scaling
		u, s, vt = jnp.linalg.svd(A, full_matrices=False)
		A = ((u @ vt) * 0.95).astype(self.param_dtype)
		B = (
			jax.random.normal(kB, (self.state_dim, self.config.input_dim), dtype=self.param_dtype)
			* (1.0 / jnp.sqrt(max(1, self.config.input_dim)))
		)
		mlp_w1 = (
			jax.random.normal(kM1, (self.state_dim, self.mlp_hidden_dim), dtype=self.param_dtype)
			* (1.0 / jnp.sqrt(self.state_dim))
		)
		mlp_b1 = jnp.zeros((self.mlp_hidden_dim,), dtype=self.param_dtype)
		mlp_w2 = (
			jax.random.normal(kM2, (self.mlp_hidden_dim, self.config.output_dim), dtype=self.param_dtype)
			* (1.0 / jnp.sqrt(self.mlp_hidden_dim))
		)
		mlp_b2 = jnp.zeros((self.config.output_dim,), dtype=self.param_dtype)
		return {"A": A, "B": B, "mlp_w1": mlp_w1, "mlp_b1": mlp_b1, "mlp_w2": mlp_w2, "mlp_b2": mlp_b2}

	def apply(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
		*,
		return_features: bool = False,
	) -> jnp.ndarray | tuple[jnp.ndarray, LRURuntimeTensors]:
		A, B = params["A"], params["B"]
		Bsz, T, D = x.shape
		x = x.astype(self.param_dtype)
		h = jnp.zeros((Bsz, self.state_dim), dtype=self.param_dtype)

		def step(h, x_t):
			pre = (h @ A.T) + (x_t @ B.T)
			z1 = jax.nn.gelu(pre @ params["mlp_w1"] + params["mlp_b1"])
			y = z1 @ params["mlp_w2"] + params["mlp_b2"]
			return pre, (y, pre)

		_, (ys, pre_ts) = jax.lax.scan(step, h, x.swapaxes(0, 1))
		output = ys.swapaxes(0, 1)
		if not return_features:
			return output
		hidden_seq = pre_ts.swapaxes(0, 1)
		features = LRURuntimeTensors(
			pre_activations=hidden_seq,
			hidden_states=hidden_seq,
			nonlinearity_jacobian_diag=jnp.ones_like(hidden_seq, dtype=self.param_dtype),
		)
		return output, features

	def analyze_batch(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
	) -> tuple[jnp.ndarray, LRURuntimeTensors, JacobianFeatureSummary]:
		outputs, tensors = self.apply(params, x, mask, return_features=True)
		A = params["A"]
		stats = compute_jacobian_features(tensors.nonlinearity_jacobian_diag, A, mask)
		return outputs, tensors, stats

	def receptive_field(self) -> int:
		# Linear recurrence has full temporal dependency
		return 10_000_000  # proxy for "unbounded" in finite computation
