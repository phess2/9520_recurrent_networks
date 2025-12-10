from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def classification_metrics(
    logits: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ll = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    nll = -jnp.sum(ll * mask) / jnp.sum(mask)
    acc = jnp.sum((jnp.argmax(logits, axis=-1) == targets) * mask) / jnp.sum(mask)
    return {"nll": nll, "accuracy": acc}


def regression_metrics(
    pred: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    mse = jnp.sum(((pred - targets) ** 2) * mask[..., None]) / jnp.sum(mask)
    nll = 0.5 * mse
    return {"nll": nll, "accuracy": 0.0 * nll}


def mutual_information_placeholder(
    latents: jnp.ndarray, inferred: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """Placeholder for MI computation between true latents and model-internal states.

    Implement later using discretization or kNN estimators.
    """
    return jnp.array(0.0)
