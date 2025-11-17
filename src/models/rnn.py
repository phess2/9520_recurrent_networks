from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from .base import BaseSequenceModel, ModelConfig

try:
	from modula.atom import Linear
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
		# Modula Linear(in_dim, out_dim) creates weights of shape [in_dim, out_dim]
		# But the einsum "...ij,...j->...i" expects [out_dim, in_dim]
		# So we swap the arguments: Linear(out_dim, in_dim) creates [out_dim, in_dim]
		self.wx = _linear_init(config.hidden_dim, config.input_dim)
		self.wh = _linear_init(config.hidden_dim, config.hidden_dim)
		self.wo = _linear_init(config.output_dim, config.hidden_dim)

	def initialize(self, key: jax.Array) -> Any:
		k1, k2, k3 = jax.random.split(key, 3)
		wx_params = self.wx.initialize(k1)
		wh_params = self.wh.initialize(k2)
		wo_params = self.wo.initialize(k3)
		return {
			"wx": wx_params,
			"wh": wh_params,
			"wo": wo_params,
		}

	def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
		B, T, D = x.shape
		h = jnp.zeros((B, self.config.hidden_dim))

		def step(carry, inputs):
			h = carry
			x_t = inputs
			a = self.wx(x_t, params["wx"]) + self.wh(h, params["wh"])
			h = jnp.tanh(a)
			y = self.wo(h, params["wo"])
			return h, y

		_, ys = jax.lax.scan(step, h, x.swapaxes(0, 1))
		return ys.swapaxes(0, 1)


class LSTM(BaseSequenceModel):
	def __init__(self, config: ModelConfig):
		super().__init__(config)
		H = config.hidden_dim
		D = config.input_dim
		# Modula Linear(in_dim, out_dim) creates weights of shape [in_dim, out_dim]
		# But the einsum "...ij,...j->...i" expects [out_dim, in_dim]
		# So we swap the arguments to match the einsum convention
		self.wxi = _linear_init(H, D)  # Input-to-hidden: want [H, D] weights
		self.whi = _linear_init(H, H)  # Hidden-to-hidden: symmetric, stays [H, H]
		self.wxf = _linear_init(H, D)
		self.whf = _linear_init(H, H)
		self.wxo = _linear_init(H, D)
		self.who = _linear_init(H, H)
		self.wxc = _linear_init(H, D)
		self.whc = _linear_init(H, H)
		self.wo = _linear_init(config.output_dim, H)  # Hidden-to-output: want [output_dim, H] weights

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
			i = jax.nn.sigmoid(self.wxi(x_t, params["wxi"]) + self.whi(h, params["whi"]))
			f = jax.nn.sigmoid(self.wxf(x_t, params["wxf"]) + self.whf(h, params["whf"]))
			o = jax.nn.sigmoid(self.wxo(x_t, params["wxo"]) + self.who(h, params["who"]))
			g = jnp.tanh(self.wxc(x_t, params["wxc"]) + self.whc(h, params["whc"]))
			c = f * c + i * g
			h = o * jnp.tanh(c)
			y = self.wo(h, params["wo"])
			return (h, c), y

		_, ys = jax.lax.scan(step, (h, c), x.swapaxes(0, 1))
		return ys.swapaxes(0, 1)
