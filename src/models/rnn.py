from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from .base import BaseSequenceModel, ModelConfig
from .nonlinearities import Nonlinearity, get_nonlinearity
from ..utils.jacobian_features import JacobianFeatureSummary, compute_jacobian_features


Array = jnp.ndarray


def _glorot_uniform(key: jax.Array, in_dim: int, out_dim: int) -> Array:
	limit = jnp.sqrt(6.0 / (in_dim + out_dim))
	return jax.random.uniform(key, (out_dim, in_dim), minval=-limit, maxval=limit)


def _init_linear(key: jax.Array, in_dim: int, out_dim: int, use_bias: bool = True) -> Dict[str, Array]:
	params: Dict[str, Array] = {"w": _glorot_uniform(key, in_dim, out_dim)}
	if use_bias:
		params["b"] = jnp.zeros((out_dim,), dtype=params["w"].dtype)
	return params


def _linear_apply(x: Array, params: Dict[str, Array]) -> Array:
	y = x @ params["w"].T
	if "b" in params:
		y = y + params["b"]
	return y


@dataclass
class RNNRuntimeTensors:
	pre_activations: Array  # [B, T, H]
	hidden_states: Array  # [B, T, H]
	nonlinearity_jacobian_diag: Array  # [B, T, H]


@dataclass
class LSTMRuntimeTensors:
	candidate_pre_activations: Array  # [B, T, H]
	candidate_states: Array  # [B, T, H]
	input_gates: Array  # [B, T, H]
	hidden_states: Array  # [B, T, H]
	nonlinearity_jacobian_diag: Array  # [B, T, H]


@dataclass
class LRURuntimeTensors:
	pre_activations: Array  # [B, T, H]
	hidden_states: Array  # [B, T, H]
	nonlinearity_jacobian_diag: Array  # [B, T, H]


class ElmanRNN(BaseSequenceModel):
	def __init__(
		self,
		config: ModelConfig,
		nonlinearity: str = "tanh",
		nonlinearity_kwargs: Dict[str, Any] | None = None,
	):
		super().__init__(config)
		self._nonlinearity: Nonlinearity = get_nonlinearity(nonlinearity, **(nonlinearity_kwargs or {}))

	def initialize(self, key: jax.Array) -> Any:
		k1, k2, k3, _ = jax.random.split(key, 4)
		return {
			"wx": _init_linear(k1, self.config.input_dim, self.config.hidden_dim, use_bias=False),
			"wh": _init_linear(k2, self.config.hidden_dim, self.config.hidden_dim, use_bias=False),
			"bias": jnp.zeros((self.config.hidden_dim,), dtype=jnp.float32),
			"wo": _init_linear(k3, self.config.hidden_dim, self.config.output_dim, use_bias=True),
		}

	def apply(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
		*,
		return_features: bool = False,
	) -> Tuple[jnp.ndarray, RNNRuntimeTensors] | jnp.ndarray:
		B, _, _ = x.shape
		H = self.config.hidden_dim
		h0 = jnp.zeros((B, H), dtype=x.dtype)
		bias = params["bias"]
		act_fn = self._nonlinearity.fn
		jac_fn = self._nonlinearity.jacobian_diag

		def step(carry, x_t):
			h_prev = carry
			pre = _linear_apply(x_t, params["wx"]) + _linear_apply(h_prev, params["wh"]) + bias
			h = act_fn(pre)
			y = _linear_apply(h, params["wo"])
			jac_diag = jac_fn(pre)
			return h, (y, pre, h, jac_diag)

		_, (ys, pre_ts, h_ts, jac_diags) = jax.lax.scan(step, h0, x.swapaxes(0, 1))
		output = ys.swapaxes(0, 1)
		if not return_features:
			return output
		features = RNNRuntimeTensors(
			pre_activations=pre_ts.swapaxes(0, 1),
			hidden_states=h_ts.swapaxes(0, 1),
			nonlinearity_jacobian_diag=jac_diags.swapaxes(0, 1),
		)
		return output, features

	def analyze_batch(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
	) -> Tuple[jnp.ndarray, RNNRuntimeTensors, JacobianFeatureSummary]:
		outputs, tensors = self.apply(params, x, mask, return_features=True)
		wh_weight = params["wh"]["w"]
		stats = compute_jacobian_features(tensors.nonlinearity_jacobian_diag, wh_weight, mask)
		return outputs, tensors, stats


class LSTM(BaseSequenceModel):
	def __init__(self, config: ModelConfig):
		super().__init__(config)
		self.hidden_dim = config.hidden_dim

	def initialize(self, key: jax.Array) -> Any:
		H = self.hidden_dim
		D = self.config.input_dim
		keys = jax.random.split(key, 9)
		return {
			"wxi": _init_linear(keys[0], D, H, use_bias=True),
			"whi": _init_linear(keys[1], H, H, use_bias=False),
			"wxf": _init_linear(keys[2], D, H, use_bias=True),
			"whf": _init_linear(keys[3], H, H, use_bias=False),
			"wxo": _init_linear(keys[4], D, H, use_bias=True),
			"who": _init_linear(keys[5], H, H, use_bias=False),
			"wxc": _init_linear(keys[6], D, H, use_bias=True),
			"whc": _init_linear(keys[7], H, H, use_bias=False),
			"wo": _init_linear(keys[8], H, self.config.output_dim, use_bias=True),
		}

	def apply(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
		*,
		return_features: bool = False,
	) -> Tuple[jnp.ndarray, LSTMRuntimeTensors] | jnp.ndarray:
		B, _, _ = x.shape
		H = self.hidden_dim
		h0 = jnp.zeros((B, H), dtype=x.dtype)
		c0 = jnp.zeros((B, H), dtype=x.dtype)
		tanh = jnp.tanh

		def step(carry, x_t):
			h_prev, c_prev = carry
			i = jax.nn.sigmoid(_linear_apply(x_t, params["wxi"]) + _linear_apply(h_prev, params["whi"]))
			f = jax.nn.sigmoid(_linear_apply(x_t, params["wxf"]) + _linear_apply(h_prev, params["whf"]))
			o = jax.nn.sigmoid(_linear_apply(x_t, params["wxo"]) + _linear_apply(h_prev, params["who"]))
			pre_g = _linear_apply(x_t, params["wxc"]) + _linear_apply(h_prev, params["whc"])
			g = tanh(pre_g)
			c = f * c_prev + i * g
			h = o * tanh(c)
			y = _linear_apply(h, params["wo"])
			jac_diag = 1.0 - g ** 2
			return (h, c), (y, pre_g, g, i, jac_diag, h)

		_, (ys, pre_gs, g_states, i_gates, jac_diags, h_states) = jax.lax.scan(step, (h0, c0), x.swapaxes(0, 1))
		output = ys.swapaxes(0, 1)
		if not return_features:
			return output
		features = LSTMRuntimeTensors(
			candidate_pre_activations=pre_gs.swapaxes(0, 1),
			candidate_states=g_states.swapaxes(0, 1),
			input_gates=i_gates.swapaxes(0, 1),
			hidden_states=h_states.swapaxes(0, 1),
			nonlinearity_jacobian_diag=jac_diags.swapaxes(0, 1),
		)
		return output, features

	def analyze_batch(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
	) -> Tuple[jnp.ndarray, LSTMRuntimeTensors, JacobianFeatureSummary]:
		outputs, tensors = self.apply(params, x, mask, return_features=True)
		# Effective Jacobian uses candidate path: diag(i_t * (1 - g_t^2)) @ W_hc
		# i_t has already scaled g_t in the cell update; include it explicitly
		hidden_diag = tensors.nonlinearity_jacobian_diag * tensors.input_gates
		wh_weight = params["whc"]["w"]
		stats = compute_jacobian_features(hidden_diag, wh_weight, mask)
		return outputs, tensors, stats


class UnitaryRNN(BaseSequenceModel):
	def __init__(
		self,
		config: ModelConfig,
		nonlinearity: str = "tanh",
		nonlinearity_kwargs: Dict[str, Any] | None = None,
	):
		super().__init__(config)
		self._nonlinearity: Nonlinearity = get_nonlinearity(nonlinearity, **(nonlinearity_kwargs or {}))

	def _orthogonal_matrix(self, raw: Array) -> Array:
		q, r = jnp.linalg.qr(raw)
		diag = jnp.sign(jnp.diag(r))
		diag = jnp.where(diag == 0.0, 1.0, diag)
		return q * diag

	def initialize(self, key: jax.Array) -> Any:
		k1, k2, k3, _ = jax.random.split(key, 4)
		return {
			"wh_raw": jax.random.normal(k1, (self.config.hidden_dim, self.config.hidden_dim)),
			"wx": _init_linear(k2, self.config.input_dim, self.config.hidden_dim, use_bias=False),
			"wo": _init_linear(k3, self.config.hidden_dim, self.config.output_dim, use_bias=True),
			"bias": jnp.zeros((self.config.hidden_dim,), dtype=jnp.float32),
		}

	def apply(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
		*,
		return_features: bool = False,
	) -> Tuple[jnp.ndarray, RNNRuntimeTensors] | jnp.ndarray:
		B, _, _ = x.shape
		H = self.config.hidden_dim
		h0 = jnp.zeros((B, H), dtype=x.dtype)
		wh = self._orthogonal_matrix(params["wh_raw"])
		act_fn = self._nonlinearity.fn
		jac_fn = self._nonlinearity.jacobian_diag
		bias = params["bias"]

		def step(carry, x_t):
			h_prev = carry
			pre = _linear_apply(x_t, params["wx"]) + (h_prev @ wh.T) + bias
			h = act_fn(pre)
			y = _linear_apply(h, params["wo"])
			jac_diag = jac_fn(pre)
			return h, (y, pre, h, jac_diag)

		_, (ys, pre_ts, h_ts, jac_diags) = jax.lax.scan(step, h0, x.swapaxes(0, 1))
		output = ys.swapaxes(0, 1)
		if not return_features:
			return output
		features = RNNRuntimeTensors(
			pre_activations=pre_ts.swapaxes(0, 1),
			hidden_states=h_ts.swapaxes(0, 1),
			nonlinearity_jacobian_diag=jac_diags.swapaxes(0, 1),
		)
		return output, features

	def analyze_batch(
		self,
		params: Any,
		x: jnp.ndarray,
		mask: jnp.ndarray,
	) -> Tuple[jnp.ndarray, RNNRuntimeTensors, JacobianFeatureSummary]:
		outputs, tensors = self.apply(params, x, mask, return_features=True)
		wh_weight = self._orthogonal_matrix(params["wh_raw"])
		stats = compute_jacobian_features(tensors.nonlinearity_jacobian_diag, wh_weight, mask)
		return outputs, tensors, stats
