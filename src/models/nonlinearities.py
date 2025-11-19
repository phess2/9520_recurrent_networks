from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import jax
import jax.numpy as jnp


Array = jnp.ndarray


@dataclass(frozen=True)
class Nonlinearity:
	name: str
	fn: Callable[[Array], Array]
	jacobian_diag: Callable[[Array], Array]


def _tanh_jacobian(x: Array) -> Array:
	h = jnp.tanh(x)
	return 1.0 - h ** 2


def _relu_jacobian(x: Array) -> Array:
	return (x > 0).astype(x.dtype)


def _leaky_relu_factory(negative_slope: float) -> Nonlinearity:
	def fn(x: Array) -> Array:
		return jnp.where(x > 0, x, negative_slope * x)

	def jacobian(x: Array) -> Array:
		return jnp.where(x > 0, jnp.ones_like(x), jnp.full_like(x, negative_slope))

	return Nonlinearity(name=f"leaky_relu_{negative_slope}", fn=fn, jacobian_diag=jacobian)


def _tanh_nonlinearity() -> Nonlinearity:
	return Nonlinearity(name="tanh", fn=jnp.tanh, jacobian_diag=_tanh_jacobian)


def _relu_nonlinearity() -> Nonlinearity:
	return Nonlinearity(name="relu", fn=jax.nn.relu, jacobian_diag=_relu_jacobian)


def _gelu_jacobian(x: Array) -> Array:
	"""Derivative of GELU with the exact formulation."""
	# From Hendrycks & Gimpel (2016)
	inv_sqrt_2pi = 0.3989422804014327
	return 0.5 * (1.0 + jax.lax.erf(x / jnp.sqrt(2.0))) + x * inv_sqrt_2pi * jnp.exp(-0.5 * x ** 2)


def _gelu_nonlinearity() -> Nonlinearity:
	return Nonlinearity(name="gelu", fn=jax.nn.gelu, jacobian_diag=_gelu_jacobian)


def _identity_nonlinearity() -> Nonlinearity:
	return Nonlinearity(name="identity", fn=lambda x: x, jacobian_diag=lambda x: jnp.ones_like(x))


_REGISTRY: Dict[str, Callable[..., Nonlinearity]] = {
	"tanh": _tanh_nonlinearity,
	"relu": _relu_nonlinearity,
	"gelu": _gelu_nonlinearity,
	"identity": _identity_nonlinearity,
}


def get_nonlinearity(name: str, **kwargs) -> Nonlinearity:
	lower = name.lower()
	if lower == "leaky_relu":
		slope = float(kwargs.get("negative_slope", 0.01))
		return _leaky_relu_factory(slope)
	builder = _REGISTRY.get(lower)
	if builder is None:
		raise ValueError(f"Unsupported nonlinearity '{name}'. Available: {sorted(_REGISTRY.keys()) + ['leaky_relu']}")
	return builder()

