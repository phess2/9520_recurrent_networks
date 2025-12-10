from __future__ import annotations

import os
from typing import Any, Callable, Dict, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..configs.schemas import FeatureEvalConfig, TrainLoopConfig
from ..data.dgm_dataset import DGMDataset, DGMConfig
from ..data.copy_dataset import CopyDataset
from ..models.base import BaseSequenceModel
from .model_factory import build_model
from .train import (
    compute_metrics,
    shift_targets,
    shift_targets_ids,
    maybe_cast_precision,
)

try:
    import orbax.checkpoint as ocp

    _ORBAX_AVAILABLE = True
except Exception:
    _ORBAX_AVAILABLE = False


def _load_params(model: BaseSequenceModel, key: jax.Array, checkpoint_path: str | None):
    if checkpoint_path:
        if not _ORBAX_AVAILABLE:
            raise RuntimeError(
                "Orbax is required to restore checkpoints but is unavailable."
            )
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(checkpoint_path)
        return restored.get("params", restored)
    return model.initialize(key)


def _build_task_and_sampler(
    task_cfg: Dict[str, Any],
) -> Tuple[Dict[str, int], Dict[str, Any], Callable[[], Dict[str, jnp.ndarray]]]:
    kind = str(task_cfg.get("kind", "dgm")).lower()
    if kind == "copy":
        num_classes = int(task_cfg["num_classes"])
        dataset = CopyDataset(
            min_lag=int(task_cfg["min_lag"]),
            max_lag=int(task_cfg["max_lag"]),
            batch_size=int(task_cfg["batch_size"]),
            num_classes=num_classes,
            seq_length=int(task_cfg.get("seq_length", 10)),
        )

        def sample_batch():
            input_ids, target_ids, mask = dataset()
            obs = jax.nn.one_hot(input_ids, num_classes, dtype=jnp.float32)
            return {"observations": obs, "targets": target_ids, "mask": mask}

        task_dims = {"input_dim": num_classes, "output_dim": num_classes}
        task_info = {"kind": "copy"}
        return task_dims, task_info, sample_batch

    data_cfg = DGMConfig(**task_cfg)
    dataset = DGMDataset(data_cfg)

    def sample_batch():
        return dataset.sample_batch()

    task_dims = {"input_dim": data_cfg.input_dim, "output_dim": data_cfg.output_dim}
    task_info = {"kind": "dgm", "data_cfg": data_cfg}
    return task_dims, task_info, sample_batch


def _evaluate_batch(
    model: BaseSequenceModel,
    params: Any,
    batch: Dict[str, jnp.ndarray],
    task_info: Dict[str, Any],
    precision: str,
):
    obs = maybe_cast_precision(batch["observations"], precision)
    mask = batch["mask"]
    kind = task_info["kind"]
    if kind == "copy":
        target_ids, mask_t = shift_targets_ids(batch["targets"], mask)
        logits = model.apply(params, obs, mask)
        metrics = compute_metrics(logits, target_ids, mask_t, discrete=True)
        return metrics, obs, mask

    data_cfg: DGMConfig = task_info["data_cfg"]
    if data_cfg.discrete_latent:
        target_ids, mask_t = shift_targets_ids(batch["obs_ids"], mask)
        logits = model.apply(params, obs, mask)
        metrics = compute_metrics(logits, target_ids, mask_t, discrete=True)
    else:
        target, mask_t = shift_targets(obs, mask)
        pred = model.apply(params, obs, mask)
        metrics = compute_metrics(pred, target, mask_t, discrete=False)
    return metrics, obs, mask


def evaluate(
    model: BaseSequenceModel,
    params: Any,
    task_info: Dict[str, Any],
    sample_batch_fn: Callable[[], Dict[str, jnp.ndarray]],
    eval_cfg: FeatureEvalConfig,
):
    os.makedirs(eval_cfg.output_dir, exist_ok=True)

    metrics_accum = {"nll": 0.0, "accuracy": 0.0}
    supports_features = hasattr(model, "analyze_batch")
    results = []

    for batch_idx in range(eval_cfg.num_batches):
        batch = sample_batch_fn()
        metrics, obs, mask = _evaluate_batch(
            model, params, batch, task_info, eval_cfg.precision
        )
        for k in metrics_accum:
            metrics_accum[k] += float(metrics.get(k, 0.0)) / eval_cfg.num_batches
        if supports_features:
            _, tensors, stats = model.analyze_batch(params, obs, mask)
            arrays_dict = {
                "frobenius_norms": stats.frobenius_norms,
                "nonlinearity_active_fraction": stats.nonlinearity_active_fraction,
                "nonlinearity_scaling": stats.nonlinearity_scaling,
                "pre_activations": tensors.pre_activations,
                "hidden_states": tensors.hidden_states,
                "lambda_max": stats.max_eigenvalue,
                "max_singular_value": stats.max_singular_value,
            }
            # Add l_eff if it was computed
            if stats.l_eff is not None:
                arrays_dict["l_eff"] = stats.l_eff
            arrays = jax.device_get(arrays_dict)
            save_path = os.path.join(eval_cfg.output_dir, f"batch_{batch_idx:04d}.npz")
            if eval_cfg.save_raw_tensors:
                np.savez(save_path, **arrays)
            results.append({k: np.array(v) for k, v in arrays.items()})

    return metrics_accum, results


@hydra.main(version_base=None, config_path="../configs", config_name="feature_eval")
def main(cfg: DictConfig) -> None:
    train_cfg = TrainLoopConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    if not isinstance(task_cfg, dict):
        raise ValueError("Task config must be a mapping.")
    task_dims, task_info, sample_batch_fn = _build_task_and_sampler(task_cfg)
    model, model_cfg = build_model(cfg.model, train_cfg, task_dims)
    eval_cfg = FeatureEvalConfig(**OmegaConf.to_container(cfg.eval, resolve=True))
    key = jax.random.PRNGKey(eval_cfg.seed)
    params = _load_params(model, key, eval_cfg.checkpoint_path)
    metrics, feature_records = evaluate(
        model, params, task_info, sample_batch_fn, eval_cfg
    )
    print("Aggregate metrics:", metrics)
    if feature_records:
        summary_path = os.path.join(eval_cfg.output_dir, "summary.npz")
        means = {
            "frobenius_mean": np.mean(
                [np.mean(rec["frobenius_norms"]) for rec in feature_records]
            ),
            "active_fraction_mean": np.mean(
                [
                    np.mean(rec["nonlinearity_active_fraction"])
                    for rec in feature_records
                ]
            ),
            "scaling_mean": np.mean(
                [np.mean(rec["nonlinearity_scaling"]) for rec in feature_records]
            ),
            "lambda_max_mean": np.mean(
                [np.mean(rec["lambda_max"]) for rec in feature_records]
            ),
            "max_singular_value_mean": np.mean(
                [np.mean(rec["max_singular_value"]) for rec in feature_records]
            ),
        }

        # Aggregate l_eff if it was computed
        # Default epsilon values used in compute_jacobian_features
        default_epsilons = (0.1, 0.3, 0.5, 0.7, 0.9)
        if feature_records and "l_eff" in feature_records[0]:
            # l_eff has shape [B, T, num_epsilons]
            # Compute mean across batches and time steps for each epsilon
            l_eff_arrays = [rec["l_eff"] for rec in feature_records]
            # Stack all batches: [total_batches*B, T, num_epsilons]
            l_eff_combined = np.concatenate(l_eff_arrays, axis=0)
            # Compute mean across batch and time dimensions: [num_epsilons]
            l_eff_means = np.mean(l_eff_combined, axis=(0, 1))

            # Add to summary with epsilon value in the key
            for eps_idx, eps_val in enumerate(default_epsilons):
                if eps_idx < len(l_eff_means):
                    means[f"l_eff_epsilon_{eps_val}_mean"] = float(l_eff_means[eps_idx])

        np.savez(summary_path, **means)
        print(f"Saved feature artifacts to {eval_cfg.output_dir}")
    else:
        print(
            "Model does not expose analyze_batch; no feature artifacts were produced."
        )


if __name__ == "__main__":
    main()
