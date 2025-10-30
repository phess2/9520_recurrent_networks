from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .base import BaseSequenceModel, ModelConfig


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

	def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
		A, B, C = params["A"], params["B"], params["C"]
		Bsz, T, D = x.shape
		h = jnp.zeros((Bsz, self.state_dim))

		def step(h, x_t):
			h = (h @ A.T) + (x_t @ B.T)
			y = h @ C.T
			return h, y

		h, ys = jax.lax.scan(step, h, x.swapaxes(0, 1))
		return ys.swapaxes(0, 1)

	def receptive_field(self) -> int:
		# Linear recurrence has full temporal dependency
		return 10_000_000  # proxy for "unbounded" in finite computation
