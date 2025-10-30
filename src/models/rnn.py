from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from .base import BaseSequenceModel, ModelConfig

try:
	from modula.atom import Linear
	from modula.bond import Tanh, Sigmoid
	_MODULA_AVAILABLE = True
except Exception:
	_MODULA_AVAILABLE = False


def _linear_init(in_dim: int, out_dim: int):
	if not _MODULA_AVAILABLE:
		raise RuntimeError("Modula not available; install from https://github.com/modula-systems/modula.git")
	lin = Linear(in_dim, out_dim)
	lin.jit()
	return lin


class ElmanRNN(BaseSequenceModel):
	def __init__(self, config: ModelConfig):
		super().__init__(config)
		self.wx = _linear_init(config.input_dim, config.hidden_dim)
		self.wh = _linear_init(config.hidden_dim, config.hidden_dim)
		self.wo = _linear_init(config.hidden_dim, config.output_dim)

	def initialize(self, key: jax.Array) -> Any:
		k1, k2, k3 = jax.random.split(key, 3)
		return {
			"wx": self.wx.initialize(k1),
			"wh": self.wh.initialize(k2),
			"wo": self.wo.initialize(k3),
		}

	def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
		B, T, D = x.shape
		h = jnp.zeros((B, self.config.hidden_dim))

		def step(carry, inputs):
			h = carry
			x_t = inputs
			a = self.wx.apply(params["wx"], x_t) + self.wh.apply(params["wh"], h)
			h = jnp.tanh(a)
			y = self.wo.apply(params["wo"], h)
			return h, y

		_, ys = jax.lax.scan(step, h, x.swapaxes(0, 1))
		return ys.swapaxes(0, 1)


class LSTM(BaseSequenceModel):
	def __init__(self, config: ModelConfig):
		super().__init__(config)
		H = config.hidden_dim
		D = config.input_dim
		self.wxi = _linear_init(D, H)
		self.whi = _linear_init(H, H)
		self.wxf = _linear_init(D, H)
		self.whf = _linear_init(H, H)
		self.wxo = _linear_init(D, H)
		self.who = _linear_init(H, H)
		self.wxc = _linear_init(D, H)
		self.whc = _linear_init(H, H)
		self.wo = _linear_init(H, config.output_dim)

	def initialize(self, key: jax.Array) -> Any:
		keys = jax.random.split(key, 9)
		return {
			"wxi": self.wxi.initialize(keys[0]), "whi": self.whi.initialize(keys[1]),
			"wxf": self.wxf.initialize(keys[2]), "whf": self.whf.initialize(keys[3]),
			"wxo": self.wxo.initialize(keys[4]), "who": self.who.initialize(keys[5]),
			"wxc": self.wxc.initialize(keys[6]), "whc": self.whc.initialize(keys[7]),
			"wo": self.wo.initialize(keys[8]),
		}

	def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
		B, T, D = x.shape
		H = self.config.hidden_dim
		h = jnp.zeros((B, H))
		c = jnp.zeros((B, H))

		def step(carry, inputs):
			h, c = carry
			x_t = inputs
			i = jax.nn.sigmoid(self.wxi.apply(params["wxi"], x_t) + self.whi.apply(params["whi"], h))
			f = jax.nn.sigmoid(self.wxf.apply(params["wxf"], x_t) + self.whf.apply(params["whf"], h))
			o = jax.nn.sigmoid(self.wxo.apply(params["wxo"], x_t) + self.who.apply(params["who"], h))
			g = jnp.tanh(self.wxc.apply(params["wxc"], x_t) + self.whc.apply(params["whc"], h))
			c = f * c + i * g
			h = o * jnp.tanh(c)
			y = self.wo.apply(params["wo"], h)
			return (h, c), y

		_, ys = jax.lax.scan(step, (h, c), x.swapaxes(0, 1))
		return ys.swapaxes(0, 1)
