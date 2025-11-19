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


class LinearRecurrentUnit(BaseSequenceModel):
	"""Minimal LRU-like linear recurrent model.

	x_{t+1} = A x_t + B u_t
	y_t = C x_t
	Where u_t is the input, y_t predicts next value/token.
	"""

	def __init__(self, config: ModelConfig):
		super().__init__(config)
		self.state_dim = config.hidden_dim

	def initialize(self, key: jax.Array) -> Any:
		kA, kB, kC = jax.random.split(key, 3)
		A = jax.random.normal(kA, (self.state_dim, self.state_dim)) * (1.0 / jnp.sqrt(self.state_dim))
		# Stabilize A via spectral norm scaling
		u, s, vt = jnp.linalg.svd(A, full_matrices=False)
		A = (u @ vt) * 0.95
		B = jax.random.normal(kB, (self.state_dim, self.config.input_dim)) * (1.0 / jnp.sqrt(max(1, self.config.input_dim)))
		C = jax.random.normal(kC, (self.config.output_dim, self.state_dim)) * (1.0 / jnp.sqrt(self.state_dim))
		return {"A": A, "B": B, "C": C}

	def apply(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
		*,
		return_features: bool = False,
	) -> jnp.ndarray | tuple[jnp.ndarray, LRURuntimeTensors]:
		A, B, C = params["A"], params["B"], params["C"]
		Bsz, T, D = x.shape
		h = jnp.zeros((Bsz, self.state_dim))

		def step(h, x_t):
			pre = (h @ A.T) + (x_t @ B.T)
			y = pre @ C.T
			return pre, (y, pre)

		_, (ys, pre_ts) = jax.lax.scan(step, h, x.swapaxes(0, 1))
		output = ys.swapaxes(0, 1)
		if not return_features:
			return output
		hidden_seq = pre_ts.swapaxes(0, 1)
		features = LRURuntimeTensors(
			pre_activations=hidden_seq,
			hidden_states=hidden_seq,
			nonlinearity_jacobian_diag=jnp.ones_like(hidden_seq),
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
