from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class JacobianFeatureSummary:
    frobenius_norms: Array  # [B, T]
    nonlinearity_active_fraction: Array  # [B, T]
    nonlinearity_scaling: Array  # [B, T, H]
    max_eigenvalue: Array  # scalar
    max_singular_value: Array  # scalar


def _max_eigenvalue(weight: Array) -> Array:
    """Compute the spectral radius (max eigenvalue magnitude) of the weight matrix."""
    weight32 = weight.astype(jnp.float32)
    eigvals = jnp.linalg.eigvals(weight32)
    spectral_radius = jnp.max(jnp.abs(eigvals))
    return spectral_radius


def _max_singular_value(weight: Array) -> Array:
    """Compute the spectral norm (largest singular value) of the weight matrix."""
    weight32 = weight.astype(jnp.float32)
    singular_values = jnp.linalg.svd(weight32, full_matrices=False, compute_uv=False)
    return jnp.max(singular_values)


def _cumulative_jacobian_frobenius(
    jac_diags: Array,
    wh_weight: Array,
    mask: Array,
) -> Array:
    """Compute ||J_t ... J_0||_F per batch/step."""
    B, T, H = jac_diags.shape
    wh = wh_weight.astype(jac_diags.dtype)
    diag_seq = jnp.swapaxes(jac_diags, 0, 1)  # [T, B, H]
    mask_seq = jnp.swapaxes(mask, 0, 1)  # [T, B]
    eye = jnp.broadcast_to(jnp.eye(H, dtype=wh.dtype), (B, H, H))

    def step(carry, inputs):
        diag_t, mask_t = inputs
        J_t = diag_t[:, :, None] * wh[None, :, :]
        J_total = jnp.einsum("bij,bjk->bik", J_t, carry)
        mask_bool = mask_t > 0.0
        J_total = jnp.where(mask_bool[:, None, None], J_total, carry)
        frob = jnp.linalg.norm(J_total, axis=(-2, -1))
        frob = jnp.where(mask_bool, frob, jnp.zeros_like(frob))
        return J_total, frob

    _, frob_seq = jax.lax.scan(step, eye, (diag_seq, mask_seq))
    return jnp.swapaxes(frob_seq, 0, 1)


def compute_jacobian_features(
    jac_diags: Array,
    wh_weight: Array,
    mask: Optional[Array] = None,
    threshold: float = 1.0 - 1e-6,
) -> JacobianFeatureSummary:
    """Aggregate Jacobian-based diagnostics for a batch."""
    if jac_diags.ndim != 3:
        raise ValueError("jac_diags must have shape [B, T, H].")
    B, T, H = jac_diags.shape
    if mask is None:
        mask = jnp.ones((B, T), dtype=jac_diags.dtype)
    if mask.shape != (B, T):
        raise ValueError(f"mask must have shape {(B, T)}, got {mask.shape}")
    mask = mask.astype(jac_diags.dtype)
    frob = _cumulative_jacobian_frobenius(jac_diags, wh_weight, mask)
    active_fraction = jnp.mean(jac_diags < threshold, axis=-1) * mask
    scaling = jac_diags * mask[:, :, None]
    lambda_max = _max_eigenvalue(wh_weight)
    sigma_max = _max_singular_value(wh_weight)
    return JacobianFeatureSummary(
        frobenius_norms=frob,
        nonlinearity_active_fraction=active_fraction,
        nonlinearity_scaling=scaling,
        max_eigenvalue=lambda_max,
        max_singular_value=sigma_max,
    )
