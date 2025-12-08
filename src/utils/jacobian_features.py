from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

Array = jnp.ndarray


@dataclass
class JacobianFeatureSummary:
    frobenius_norms: Array
    nonlinearity_active_fraction: Array
    nonlinearity_scaling: Array
    max_eigenvalue: Array
    max_singular_value: Array
    l_eff: Optional[Array] = None


def _max_eigenvalue(weight: Array) -> Array:
    weight32 = weight.astype(jnp.float32)
    eigvals = jnp.linalg.eigvals(weight32)
    return jnp.max(jnp.abs(eigvals))


def _max_singular_value(weight: Array) -> Array:
    weight32 = weight.astype(jnp.float32)
    singular_values = jnp.linalg.svd(weight32, full_matrices=False, compute_uv=False)
    return jnp.max(singular_values)


def _cumulative_jacobian_frobenius(
    jac_diags: Array,
    wh_weight: Array,
    mask: Array,
) -> Array:
    B, T, H = jac_diags.shape
    wh = wh_weight.astype(jac_diags.dtype)
    diag_seq = jnp.swapaxes(jac_diags, 0, 1)
    mask_seq = jnp.swapaxes(mask, 0, 1)
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


def _jacobian_lookback_frobenius(
    jac_diags: Array,
    wh_weight: Array,
    mask: Array,
) -> Array:
    B, T, H = jac_diags.shape
    wh = wh_weight.astype(jac_diags.dtype)
    diag_seq = jnp.swapaxes(jac_diags, 0, 1)
    mask_seq = jnp.swapaxes(mask, 0, 1)

    identity_norm = jnp.sqrt(jnp.array(H, dtype=wh.dtype))
    eye = jnp.broadcast_to(jnp.eye(H, dtype=wh.dtype), (B, H, H))
    all_norms = jnp.zeros((B, T, T + 1), dtype=wh.dtype)

    def scan_step(carry, inputs):
        diag_t, mask_t, t_idx = inputs
        jacobian_history, norms = carry

        J_t = diag_t[:, :, None] * wh[None, :, :]
        mask_bool = mask_t > 0.0

        jacobian_history = jnp.concatenate(
            [J_t[:, None, :, :], jacobian_history[:, :-1, :, :]], axis=1
        )

        time_step_norms = jnp.zeros((B, T + 1), dtype=wh.dtype)
        time_step_norms = time_step_norms.at[:, 0].set(identity_norm)

        def compute_lag_norm(lag, carry_state):
            J_cumulative, norms_array = carry_state
            valid_lag = lag <= (t_idx + 1)
            hist_idx = lag - 1
            J_hist = jacobian_history[:, hist_idx, :, :]
            J_cumulative = jnp.einsum("bij,bjk->bik", J_hist, J_cumulative)
            J_cumulative = jnp.where(mask_bool[:, None, None], J_cumulative, eye)
            frob = jnp.linalg.norm(J_cumulative, axis=(-2, -1))
            frob = jnp.where(mask_bool, frob, jnp.zeros_like(frob))
            updated_norms = norms_array.at[:, lag].set(frob)
            norms_array = jnp.where(valid_lag, updated_norms, norms_array)
            return (J_cumulative, norms_array)

        initial_state = (eye, time_step_norms)
        final_state = jax.lax.fori_loop(1, T + 1, compute_lag_norm, initial_state)
        _, time_step_norms = final_state

        norms = norms.at[:, t_idx, :].set(time_step_norms)

        return (jacobian_history, norms), None

    jacobian_history = jnp.zeros((B, T, H, H), dtype=wh.dtype)
    initial_carry = (jacobian_history, all_norms)

    time_indices = jnp.arange(T)
    inputs = (diag_seq, mask_seq, time_indices)

    final_carry, _ = jax.lax.scan(scan_step, initial_carry, inputs)
    _, final_norms = final_carry

    return final_norms


def _compute_l_eff(
    lookback_norms: Array,
    epsilon_values: Tuple[float, ...],
    mask: Array,
) -> Array:
    B, T, max_lag = lookback_norms.shape

    epsilons = jnp.array(epsilon_values, dtype=lookback_norms.dtype)

    lookback_norms_expanded = lookback_norms[:, :, :, None]
    epsilons_expanded = epsilons[None, None, None, :]

    greater_than_epsilon = lookback_norms_expanded > epsilons_expanded

    lag_indices = jnp.arange(max_lag, dtype=lookback_norms.dtype)
    lag_indices_expanded = lag_indices[None, None, :, None]

    valid_lags = jnp.where(
        greater_than_epsilon, lag_indices_expanded, -1.0
    )

    l_eff = jnp.max(valid_lags, axis=2)
    l_eff = jnp.maximum(l_eff, 0.0)

    mask_expanded = mask[:, :, None]
    l_eff = l_eff * mask_expanded

    return l_eff


def compute_jacobian_features(
    jac_diags: Array,
    wh_weight: Array,
    mask: Optional[Array] = None,
    threshold: float = 1.0 - 1e-6,
    epsilon_values: Optional[Tuple[float, ...]] = (0.1, 0.25, 0.5, 1.0, 2.0, 10.0),
) -> JacobianFeatureSummary:
    B, T, H = jac_diags.shape
    if mask is None:
        mask = jnp.ones((B, T), dtype=jac_diags.dtype)
    mask = mask.astype(jac_diags.dtype)
    frob = _cumulative_jacobian_frobenius(jac_diags, wh_weight, mask)
    active_fraction = jnp.mean(jac_diags < threshold, axis=-1) * mask
    scaling = jac_diags * mask[:, :, None]
    lambda_max = _max_eigenvalue(wh_weight)
    sigma_max = _max_singular_value(wh_weight)

    l_eff = None
    if epsilon_values is not None:
        lookback_norms = _jacobian_lookback_frobenius(jac_diags, wh_weight, mask)
        l_eff = _compute_l_eff(lookback_norms, epsilon_values, mask)

    return JacobianFeatureSummary(
        frobenius_norms=frob,
        nonlinearity_active_fraction=active_fraction,
        nonlinearity_scaling=scaling,
        max_eigenvalue=lambda_max,
        max_singular_value=sigma_max,
        l_eff=l_eff,
    )
