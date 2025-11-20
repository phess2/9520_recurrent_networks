from __future__ import annotations

import os
import re
from dataclasses import asdict
from typing import Dict, Any
import time

import hydra
import jax
import jax.numpy as jnp
import optax
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf

from ..configs.schemas import OptimizerConfig, TrainLoopConfig
from ..data.dgm_dataset import DGMDataset, DGMConfig
from ..models.base import BaseSequenceModel, ModelConfig
from .model_factory import build_model
import orbax.checkpoint as ocp

from ..utils.metrics import mutual_information_placeholder


def _validate_precisions(model_cfg: ModelConfig, train_cfg: TrainLoopConfig) -> None:
    if model_cfg.precision != train_cfg.precision:
        raise ValueError(
            f"model.precision ({model_cfg.precision}) must match train.precision ({train_cfg.precision})."
        )
    param_dtype = model_cfg.param_dtype or model_cfg.precision
    if param_dtype != train_cfg.precision:
        raise ValueError(
            f"model.param_dtype ({param_dtype}) must match train.precision ({train_cfg.precision})."
        )


def _format_hms(seconds: float) -> str:
    if not jnp.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_optimizer(
    cfg: OptimizerConfig, total_steps: int
) -> optax.GradientTransformation:
    if cfg.scheduler == "linear":
        schedule = optax.linear_schedule(
            init_value=0.0, end_value=cfg.lr, transition_steps=cfg.warmup_steps
        )
    elif cfg.scheduler == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=max(total_steps - cfg.warmup_steps, 1),
            end_value=0.0,
        )
    elif cfg.scheduler == "none":
        schedule = cfg.lr
    else:
        raise ValueError(f"Unknown scheduler {cfg.scheduler}")

    name = cfg.name.lower()
    if name == "adamw":
        return optax.adamw(schedule, weight_decay=cfg.weight_decay)
    elif name == "sgd":
        return optax.sgd(schedule, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {cfg.name}")


def compute_metrics(
    pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray, discrete: bool
) -> Dict[str, jnp.ndarray]:
    if discrete:
        logits = pred
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ll = jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]
        nll = -jnp.sum(ll * mask) / jnp.sum(mask)
        acc = jnp.sum((jnp.argmax(logits, axis=-1) == target) * mask) / jnp.sum(mask)
        return {"nll": nll, "accuracy": acc}
    else:
        mse = jnp.sum(((pred - target) ** 2) * mask[..., None]) / jnp.sum(mask)
        nll = 0.5 * mse
        return {"nll": nll, "accuracy": 0.0 * nll}


def shift_targets(observations: jnp.ndarray, mask: jnp.ndarray):
    x = observations
    x_next = jnp.concatenate([x[:, 1:], jnp.zeros_like(x[:, :1])], axis=1)
    mask_next = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
    return x_next, mask_next


def shift_targets_ids(obs_ids: jnp.ndarray, mask: jnp.ndarray):
    ids_next = jnp.concatenate([obs_ids[:, 1:], jnp.zeros_like(obs_ids[:, :1])], axis=1)
    mask_next = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
    return ids_next, mask_next


def maybe_cast_precision(x: jnp.ndarray, precision: str) -> jnp.ndarray:
    if precision == "bfloat16":
        return x.astype(jnp.bfloat16)
    return x


def train(
    model: BaseSequenceModel,
    model_cfg: ModelConfig,
    data_cfg: DGMConfig,
    train_cfg: TrainLoopConfig,
    optimizer_cfg: OptimizerConfig,
) -> None:
    _validate_precisions(model_cfg, train_cfg)
    if not train_cfg.entity:
        raise ValueError("train.entity must be set to the target W&B team.")
    if train_cfg.wandb_api_key:
        wandb.login(key=train_cfg.wandb_api_key, relogin=True)
    wandb.init(
        project=train_cfg.project,
        name=train_cfg.run_name,
        entity=train_cfg.entity,
        config={
            "model": model_cfg.__dict__,
            "data": data_cfg.__dict__,
            "train": asdict(train_cfg),
            "optimizer": asdict(optimizer_cfg),
        },
    )

    dataset = DGMDataset(data_cfg)
    opt = build_optimizer(optimizer_cfg, train_cfg.steps)

    key = jax.random.PRNGKey(train_cfg.seed)
    params = model.initialize(key)
    opt_state = opt.init(params)
    supports_feature_logging = hasattr(model, "analyze_batch")
    feature_log_every = int(getattr(train_cfg, "feature_log_every", 0) or 0)
    feature_log_batches = max(1, int(getattr(train_cfg, "feature_log_max_batches", 1)))
    feature_save_dir = train_cfg.feature_save_dir
    if feature_save_dir:
        os.makedirs(feature_save_dir, exist_ok=True)
    last_feature_metrics: Dict[str, float] | None = None
    last_eval_metrics: Dict[str, float] | None = None
    start_time = time.time()

    ckpt_mgr = None
    checkpoint_directory = os.path.join("checkpoints", train_cfg.project, train_cfg.run_name)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_directory, ocp.PyTreeCheckpointer())

    @jax.jit
    def step_continuous(params, opt_state, batch):
        obs = batch["observations"]
        mask = batch["mask"]
        target, mask_t = shift_targets(obs, mask)

        def loss_fn(p):
            pred = model.apply(p, maybe_cast_precision(obs, train_cfg.precision), mask)
            metrics = compute_metrics(pred, target, mask_t, discrete=False)
            return metrics["nll"], metrics

        (nll, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def step_discrete(params, opt_state, batch):
        obs = batch["observations"]
        obs_ids = batch["obs_ids"]
        mask = batch["mask"]
        target_ids, mask_t = shift_targets_ids(obs_ids, mask)

        def loss_fn(p):
            logits = model.apply(
                p, maybe_cast_precision(obs, train_cfg.precision), mask
            )
            metrics = compute_metrics(logits, target_ids, mask_t, discrete=True)
            return metrics["nll"], metrics

        (nll, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_step_continuous(p, batch):
        obs = batch["observations"]
        mask = batch["mask"]
        target, mask_t = shift_targets(obs, mask)
        pred = model.apply(p, maybe_cast_precision(obs, train_cfg.precision), mask)
        metrics = compute_metrics(pred, target, mask_t, discrete=False)
        return metrics

    @jax.jit
    def eval_step_discrete(p, batch):
        obs = batch["observations"]
        obs_ids = batch["obs_ids"]
        mask = batch["mask"]
        target_ids, mask_t = shift_targets_ids(obs_ids, mask)
        logits = model.apply(p, maybe_cast_precision(obs, train_cfg.precision), mask)
        metrics = compute_metrics(logits, target_ids, mask_t, discrete=True)
        return metrics

    best_metric = None
    best_is_higher = train_cfg.ckpt_metric.lower() == "accuracy"

    def maybe_log_features(step_idx: int, params, batch: Dict[str, jnp.ndarray]):
        if feature_log_every <= 0 or (step_idx % feature_log_every) != 0:
            return None
        if not supports_feature_logging:
            return None
        mask = batch["mask"]
        mask_sum = 0.0
        frob_sum = 0.0
        active_sum = 0.0
        scaling_sum = 0.0
        scaling_count = 0.0
        lambda_max_value = None
        sigma_max_value = None
        payload_to_save = None

        def process_batch(curr_batch):
            obs = maybe_cast_precision(curr_batch["observations"], train_cfg.precision)
            curr_mask = curr_batch["mask"]
            _, tensors, stats = model.analyze_batch(params, obs, curr_mask)
            return tensors, stats, curr_mask

        for idx in range(feature_log_batches):
            curr_batch = batch if idx == 0 else dataset.sample_batch()
            tensors, stats, curr_mask = process_batch(curr_batch)
            curr_mask = curr_mask.astype(jnp.float32)
            mask_sum = mask_sum + float(jnp.sum(curr_mask))
            frob_sum = frob_sum + float(jnp.sum(stats.frobenius_norms * curr_mask))
            active_sum = active_sum + float(jnp.sum(stats.nonlinearity_active_fraction))
            scaling_sum = scaling_sum + float(jnp.sum(stats.nonlinearity_scaling))
            scaling_count = scaling_count + float(
                jnp.sum(curr_mask) * stats.nonlinearity_scaling.shape[-1]
            )
            if lambda_max_value is None:
                lambda_max_value = float(stats.max_eigenvalue)
            if sigma_max_value is None:
                sigma_max_value = float(stats.max_singular_value)
            if payload_to_save is None and feature_save_dir:
                payload_to_save = {
                    "frobenius_norms": stats.frobenius_norms,
                    "active_fraction": stats.nonlinearity_active_fraction,
                    "scaling": stats.nonlinearity_scaling,
                    "pre_activations": tensors.pre_activations,
                    "hidden_states": tensors.hidden_states,
                    "lambda_max": stats.max_eigenvalue,
                    "max_singular_value": stats.max_singular_value,
                }

        if mask_sum <= 0:
            return None
        eps = 1e-6
        feature_metrics = {
            "features/frobenius_mean": frob_sum / (mask_sum + eps),
            "features/nonlinearity_active_fraction": active_sum / (mask_sum + eps),
            "features/nonlinearity_scale_mean": scaling_sum / (scaling_count + eps),
        }
        if lambda_max_value is not None:
            feature_metrics["features/lambda_max"] = lambda_max_value
        if sigma_max_value is not None:
            feature_metrics["features/max_singular_value"] = sigma_max_value
        wandb.log({"step": step_idx, **feature_metrics})
        if payload_to_save is not None:
            save_path = os.path.join(feature_save_dir, f"step_{step_idx:06d}.npz")
            payload_np = jax.device_get(payload_to_save)
            np.savez(save_path, **payload_np)
        return feature_metrics

    def maybe_save_best(step_idx: int, metrics: Dict[str, Any], params, opt_state):
        now = float(metrics[train_cfg.ckpt_metric])
        nonlocal best_metric
        improved = (best_metric is None) or (
            (now > best_metric) if best_is_higher else (now < best_metric)
        )
        if improved:
            best_metric = now
            wandb.run.summary["best_" + train_cfg.ckpt_metric] = now
            if train_cfg.save_best and ckpt_mgr is not None:
                ckpt_mgr.save(
                    step_idx,
                    args={
                        "params": params,
                        "opt_state": opt_state,
                        "best_metric": best_metric,
                    },
                )

    for step_idx in range(1, train_cfg.steps + 1):
        batch = dataset.sample_batch()
        if data_cfg.discrete_latent:
            params, opt_state, metrics = step_discrete(params, opt_state, batch)
        else:
            params, opt_state, metrics = step_continuous(params, opt_state, batch)

        # Mutual information placeholder (no internal states available yet)
        mi = mutual_information_placeholder(
            batch.get("latents"), batch.get("latents"), batch["mask"]
        )  # placeholder

        if step_idx % train_cfg.log_every == 0:
            train_stats = {k: float(v) for k, v in metrics.items()}
            wandb.log({"step": step_idx, **train_stats, "mi_placeholder": float(mi)})
            feature_stats = maybe_log_features(step_idx, params, batch)
            if feature_stats is not None:
                last_feature_metrics = feature_stats

            elapsed = time.time() - start_time
            speed = elapsed / step_idx if step_idx > 0 else float("inf")
            eta = (train_cfg.steps - step_idx) * speed if step_idx > 0 else float("inf")
            print(
                f"\nStep {step_idx}/{train_cfg.steps}  ({_format_hms(elapsed)} elapsed, {_format_hms(eta)} ETA)"
            )
            print(
                "  train ┆ "
                + " │ ".join(
                    f"{name}: {value:.4f}" for name, value in train_stats.items()
                )
            )
            if last_eval_metrics:
                print(
                    "  eval  ┆ "
                    + " │ ".join(
                        f"{name}: {value:.4f}"
                        for name, value in last_eval_metrics.items()
                    )
                )
            if last_feature_metrics:
                print(
                    "  feat  ┆ "
                    + " │ ".join(
                        f"{name.split('/')[-1]}: {value:.4f}"
                        for name, value in last_feature_metrics.items()
                    )
                )
            print("-" * 80)
        else:
            maybe_log_features(step_idx, params, batch)

        if step_idx % train_cfg.eval_every == 0:
            # Evaluate over eval_steps mini-batches
            agg = {"nll": 0.0, "accuracy": 0.0}
            for _ in range(train_cfg.eval_steps):
                b = dataset.sample_batch()
                m = (
                    eval_step_discrete(params, b)
                    if data_cfg.discrete_latent
                    else eval_step_continuous(params, b)
                )
                agg["nll"] += float(m["nll"]) / train_cfg.eval_steps
                agg["accuracy"] += float(m["accuracy"]) / train_cfg.eval_steps
            wandb.log(
                {
                    "step": step_idx,
                    "eval/nll": agg["nll"],
                    "eval/accuracy": agg["accuracy"],
                }
            )
            metrics_for_ckpt = {train_cfg.ckpt_metric: agg[train_cfg.ckpt_metric]}
            maybe_save_best(step_idx, metrics_for_ckpt, params, opt_state)
            last_eval_metrics = {k: float(v) for k, v in agg.items()}

        if step_idx % train_cfg.ckpt_every == 0 and ckpt_mgr is not None:
            ckpt_mgr.save(step_idx, args={"params": params, "opt_state": opt_state})

    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="dgm")
def main(cfg: DictConfig) -> None:
    train_cfg = TrainLoopConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    optimizer_cfg = OptimizerConfig(
        **OmegaConf.to_container(cfg.optimizer, resolve=True)
    )
    data_cfg = DGMConfig(**OmegaConf.to_container(cfg.task, resolve=True))
    task_dims = {"input_dim": data_cfg.input_dim, "output_dim": data_cfg.output_dim}
    model, model_cfg = build_model(cfg.model, train_cfg, task_dims)
    train(model, model_cfg, data_cfg, train_cfg, optimizer_cfg)


if __name__ == "__main__":
    main()
