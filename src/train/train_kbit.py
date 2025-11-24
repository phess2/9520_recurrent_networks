from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

import wandb

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..configs.schemas import OptimizerConfig, TrainLoopConfig
from ..data.kbit_flipflop import KBitFlipFlopDataset
from .model_factory import build_model
from .train_base import (
    build_optimizer,
    configure_orbax_logging,
    create_checkpoint_manager,
    ensure_prediction_artifacts,
    format_hms,
    initialize_wandb_run,
    maybe_cast_precision,
    save_weight_checkpoint,
    validate_precisions,
)

configure_orbax_logging()


def compute_regression_metrics(
    predictions: jnp.ndarray, targets: jnp.ndarray, mask: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    mask = mask.astype(jnp.float32)[..., None]
    error = predictions - targets
    squared_error = error**2
    abs_error = jnp.abs(error)
    denom = jnp.maximum(jnp.sum(mask), 1e-6)
    mse = jnp.sum(squared_error * mask) / denom
    mae = jnp.sum(abs_error * mask) / denom
    return {"mse": mse, "mae": mae, "nll": 0.5 * mse}


def train_kbit(
    task_cfg: Dict[str, Any],
    model_cfg: DictConfig,
    train_cfg: TrainLoopConfig,
    optimizer_cfg: OptimizerConfig,
) -> None:
    run_identifier, hydra_run_dir = initialize_wandb_run(
        train_cfg, task_cfg, model_cfg, optimizer_cfg
    )

    dataset = KBitFlipFlopDataset(
        k=int(task_cfg["k"]),
        batch_size=int(task_cfg["batch_size"]),
        seq_length=int(task_cfg["seq_length"]),
        p=float(task_cfg.get("p", 0.01)),
        noise_std=float(task_cfg.get("noise_std", 0.01)),
    )
    full_length_dataset = KBitFlipFlopDataset(
        k=int(task_cfg["k"]),
        batch_size=int(task_cfg["batch_size"]),
        seq_length=int(task_cfg["seq_length"]),
        p=float(task_cfg.get("p", 0.01)),
        noise_std=float(task_cfg.get("noise_std", 0.01)),
    )
    noise_generalization_stddevs = tuple(
        float(value)
        for value in (train_cfg.noise_generalization_stddevs or ())
        if value is not None
    )
    generalization_rng = jax.random.PRNGKey(train_cfg.seed + 3)

    input_dim = output_dim = int(task_cfg["k"])
    task_dims = {"input_dim": input_dim, "output_dim": output_dim}
    model, model_config = build_model(model_cfg, train_cfg, task_dims)
    validate_precisions(model_config, train_cfg)

    optimizer = build_optimizer(optimizer_cfg, train_cfg.steps)

    random_key = jax.random.PRNGKey(train_cfg.seed)
    model_params = model.initialize(random_key)
    optimizer_state = optimizer.init(model_params)
    supports_feature_logging = hasattr(model, "analyze_batch")
    feature_log_every = int(getattr(train_cfg, "feature_log_every", 0) or 0)
    feature_log_batches = max(1, int(getattr(train_cfg, "feature_log_max_batches", 1)))
    feature_save_dir = train_cfg.feature_save_dir
    if feature_save_dir:
        os.makedirs(feature_save_dir, exist_ok=True)
    last_feature_metrics: Dict[str, float] | None = None
    last_eval_metrics: Dict[str, float] | None = None
    last_generalization_metrics: Dict[str, float] | None = None
    start_time = time.time()

    dataset_name = "kbit_flipflop"
    architecture_name = re.sub(
        r"(?<!^)(?=[A-Z])", "_", model.__class__.__name__
    ).lower()
    checkpoint_manager = create_checkpoint_manager(
        train_cfg, dataset_name, architecture_name
    )
    prediction_plot_dir, jacobian_plot_dir = ensure_prediction_artifacts(
        train_cfg, hydra_run_dir, dataset_name, architecture_name, run_identifier
    )
    jacobian_eval_history: List[Dict[str, Any]] = []

    def cast_inputs(arr: jnp.ndarray) -> jnp.ndarray:
        return maybe_cast_precision(arr, train_cfg.precision)

    def evaluate_with_noise(
        params_to_eval,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        noise_std: float = 0.0,
        rng_key: jax.Array | None = None,
    ) -> Dict[str, jnp.ndarray]:
        processed_inputs = cast_inputs(inputs)
        if noise_std and noise_std > 0.0:
            if rng_key is None:
                rng_key = jax.random.PRNGKey(0)
            noise = noise_std * jax.random.normal(
                rng_key, processed_inputs.shape, dtype=processed_inputs.dtype
            )
            processed_inputs = processed_inputs + noise
        predictions = model.apply(params_to_eval, processed_inputs, mask)
        return compute_regression_metrics(predictions, targets, mask)

    def run_length_generalization(step_index: int, params_for_eval) -> Dict[str, float]:
        logged_metrics: Dict[str, float] = {}
        inputs, targets, mask = full_length_dataset()
        metrics = evaluate_with_noise(params_for_eval, inputs, targets, mask, 0.0)
        log_prefix = "features/generalization/full_length"
        payload = {
            "step": step_index,
            f"{log_prefix}/mse": float(metrics["mse"]),
            f"{log_prefix}/mae": float(metrics["mae"]),
        }
        wandb.log(payload)
        logged_metrics.update({k: v for k, v in payload.items() if k != "step"})
        return logged_metrics

    def run_noise_generalization(step_index: int, params_for_eval) -> Dict[str, float]:
        logged_metrics: Dict[str, float] = {}
        if not noise_generalization_stddevs:
            return logged_metrics
        nonlocal generalization_rng
        for std in noise_generalization_stddevs:
            if std is None or std < 0.0:
                continue
            generalization_rng, noise_key = jax.random.split(generalization_rng)
            inputs, targets, mask = dataset()
            metrics = evaluate_with_noise(
                params_for_eval, inputs, targets, mask, float(std), noise_key
            )
            log_prefix = f"features/generalization/noise_{float(std):.3f}"
            payload = {
                "step": step_index,
                f"{log_prefix}/mse": float(metrics["mse"]),
                f"{log_prefix}/mae": float(metrics["mae"]),
            }
            wandb.log(payload)
            logged_metrics.update({k: v for k, v in payload.items() if k != "step"})
        return logged_metrics

    @jax.jit
    def train_step(model_params, optimizer_state, inputs, targets, mask):
        def loss_fn(current_params):
            predictions = model.apply(current_params, cast_inputs(inputs), mask)
            metrics = compute_regression_metrics(predictions, targets, mask)
            return metrics["nll"], metrics

        (negative_log_likelihood, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(model_params)
        parameter_updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, model_params
        )
        model_params = optax.apply_updates(model_params, parameter_updates)
        return model_params, optimizer_state, metrics

    @jax.jit
    def eval_step(model_params, inputs, targets, mask):
        predictions = model.apply(model_params, cast_inputs(inputs), mask)
        metrics = compute_regression_metrics(predictions, targets, mask)
        return metrics

    best_metric_value = None
    higher_is_better = train_cfg.ckpt_metric.lower() == "accuracy"

    def maybe_save_best(
        step_index: int, metrics: Dict[str, Any], model_params, optimizer_state
    ):
        current_metric_value = float(metrics[train_cfg.ckpt_metric])
        nonlocal best_metric_value
        if best_metric_value is None:
            is_improvement = True
        else:
            is_improvement = (
                (current_metric_value > best_metric_value)
                if higher_is_better
                else (current_metric_value < best_metric_value)
            )
        if is_improvement:
            best_metric_value = current_metric_value
            wandb.run.summary["best_" + train_cfg.ckpt_metric] = current_metric_value
            if (
                train_cfg.save_best
                and not train_cfg.disable_checkpointing
                and not train_cfg.sweep_run
                and checkpoint_manager is not None
            ):
                checkpoint_manager.save(
                    step_index,
                    args=ocp.args.PyTreeSave(
                        {
                            "params": model_params,
                            "opt_state": optimizer_state,
                            "best_metric": best_metric_value,
                        }
                    ),
                )

    def maybe_log_features(
        step_index: int,
        model_params,
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        if train_cfg.sweep_run:
            return None
        if feature_log_every <= 0 or (step_index % feature_log_every) != 0:
            return None
        if not supports_feature_logging:
            return None

        mask_total = 0.0
        frob_sum = 0.0
        active_sum = 0.0
        scaling_sum = 0.0
        scaling_count = 0.0
        lambda_max_value = None
        sigma_max_value = None
        payload_to_save = None

        def process(batch_inputs, batch_mask):
            embedded = cast_inputs(batch_inputs)
            _, tensors, stats = model.analyze_batch(model_params, embedded, batch_mask)
            return tensors, stats

        for idx in range(feature_log_batches):
            if idx == 0:
                batch_inputs = inputs
                batch_mask = mask
            else:
                batch_inputs, _, batch_mask = dataset()
            tensors, stats = process(batch_inputs, batch_mask)
            batch_mask = batch_mask.astype(jnp.float32)
            mask_total += float(jnp.sum(batch_mask))
            frob_sum += float(jnp.sum(stats.frobenius_norms * batch_mask))
            active_sum += float(jnp.sum(stats.nonlinearity_active_fraction))
            scaling_sum += float(jnp.sum(stats.nonlinearity_scaling))
            scaling_count += float(
                jnp.sum(batch_mask) * stats.nonlinearity_scaling.shape[-1]
            )
            if lambda_max_value is None:
                lambda_max_value = float(stats.max_eigenvalue)
            if sigma_max_value is None:
                sigma_max_value = float(stats.max_singular_value)
            if payload_to_save is None and feature_save_dir:
                payload = {
                    "frobenius_norms": stats.frobenius_norms,
                    "active_fraction": stats.nonlinearity_active_fraction,
                    "scaling": stats.nonlinearity_scaling,
                }
                if hasattr(tensors, "pre_activations"):
                    payload["pre_activations"] = tensors.pre_activations
                if hasattr(tensors, "candidate_pre_activations"):
                    payload["candidate_pre_activations"] = (
                        tensors.candidate_pre_activations
                    )
                if hasattr(tensors, "hidden_states"):
                    payload["hidden_states"] = tensors.hidden_states
                payload["lambda_max"] = stats.max_eigenvalue
                payload["max_singular_value"] = stats.max_singular_value
                payload_to_save = payload

        if mask_total <= 0:
            return None
        eps = 1e-6
        feature_metrics = {
            "features/frobenius_mean": frob_sum / (mask_total + eps),
            "features/nonlinearity_active_fraction": active_sum / (mask_total + eps),
            "features/nonlinearity_scale_mean": scaling_sum / (scaling_count + eps),
        }
        if lambda_max_value is not None:
            feature_metrics["features/lambda_max"] = lambda_max_value
        if sigma_max_value is not None:
            feature_metrics["features/max_singular_value"] = sigma_max_value
        wandb.log({"step": step_index, **feature_metrics})
        if payload_to_save is not None and feature_save_dir:
            save_path = os.path.join(feature_save_dir, f"step_{step_index:06d}.npz")
            payload_np = jax.device_get(payload_to_save)
            np.savez(save_path, **payload_np)
        return feature_metrics

    def create_prediction_figure(
        model_params,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        example_idx: int = 0,
    ):
        predictions = model.apply(model_params, cast_inputs(inputs), mask)
        inputs_np = np.asarray(jax.device_get(inputs))
        targets_np = np.asarray(jax.device_get(targets))
        preds_np = np.asarray(jax.device_get(predictions))
        mask_np = np.asarray(jax.device_get(mask)).astype(bool)

        example_idx = int(np.clip(example_idx, 0, inputs_np.shape[0] - 1))
        inp = inputs_np[example_idx]
        tgt = targets_np[example_idx]
        pred = preds_np[example_idx]
        valid = mask_np[example_idx]

        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)
        cmap = "bwr"
        vlim = max(1.0, float(np.max(np.abs(inp))))
        im0 = axes[0].imshow(inp.T, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim)
        axes[0].set_ylabel("input bit")
        axes[0].set_title("Inputs")
        im1 = axes[1].imshow(tgt.T, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
        axes[1].set_ylabel("target bit")
        axes[1].set_title("Targets")
        im2 = axes[2].imshow(pred.T, aspect="auto", cmap=cmap, vmin=-1.0, vmax=1.0)
        axes[2].set_ylabel("pred bit")
        axes[2].set_title("Predictions")
        axes[2].set_xlabel("time step")
        for ax in axes:
            ax.set_yticks(range(inp.shape[-1]))
            ax.grid(False)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()

        mse = float(np.mean(((pred - tgt)[valid]) ** 2))
        return fig, mse

    def log_prediction_figure(
        step_index: int,
        model_params,
        input_ids: jnp.ndarray,
        target_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ):
        if prediction_plot_dir is None:
            return
        try:
            fig, mse = create_prediction_figure(
                model_params, input_ids, target_ids, attention_mask
            )
        except Exception:
            logging.exception("Failed to create prediction visualization.")
            return

        caption = f"Step {step_index} - Sequence MSE {mse:.4f}"
        try:
            wandb.run.summary["prediction_plot"] = wandb.Image(fig, caption=caption)
        except Exception:
            logging.exception(
                "Failed to log prediction visualization to wandb summary."
            )
        figure_path = os.path.join(
            prediction_plot_dir, f"prediction_step_{step_index:06d}.png"
        )
        fig.savefig(figure_path)
        plt.close(fig)

    def compute_jacobian_stats(
        model_params,
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = cast_inputs(inputs)
        full_mask = jnp.ones_like(mask)
        _, tensors, stats = model.analyze_batch(model_params, embedded, full_mask)
        frob = np.asarray(jax.device_get(stats.frobenius_norms))
        if frob.ndim != 2:
            raise ValueError("Expected frobenius norms with shape [B, T].")
        mean = np.mean(frob, axis=0)
        std = np.std(frob, axis=0)
        time_points = np.arange(frob.shape[1])
        return time_points, mean, std

    def log_jacobian_figure(
        step_index: int,
        model_params,
        inputs: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        if not supports_feature_logging or jacobian_plot_dir is None:
            return
        try:
            time_points, mean, std = compute_jacobian_stats(model_params, inputs, mask)
        except Exception:
            logging.exception("Failed to compute jacobian statistics.")
            return

        jacobian_eval_history.append(
            {
                "step": int(step_index),
                "time": time_points.tolist(),
                "mean": mean.tolist(),
                "std": std.tolist(),
            }
        )
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time_points, mean, label=f"Step {step_index}")
        ax.fill_between(time_points, mean - std, mean + std, alpha=0.3)
        ax.set_title("Jacobian Frobenius Norm Across Sequence")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Frobenius norm")
        ax.legend(loc="upper right")
        ax.set_yscale("log")
        upper = float(np.max(mean + std) + 1e-6)
        lower = float(np.min(np.maximum(mean - std, 1e-8)))
        lower = max(lower, 1e-8)
        ax.set_ylim(lower, upper)
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        figure_path = os.path.join(
            jacobian_plot_dir, f"jacobian_step_{step_index:06d}.png"
        )
        fig.savefig(figure_path)
        try:
            wandb.log(
                {
                    "step": step_index,
                    "eval/jacobian_plot": wandb.Image(
                        fig, caption=f"Jacobian norms at step {step_index}"
                    ),
                }
            )
        except Exception:
            logging.exception("Failed to log jacobian plot to wandb.")
        plt.close(fig)

    def log_final_jacobian_summary():
        if not jacobian_eval_history or jacobian_plot_dir is None:
            return
        summary_path = os.path.join(jacobian_plot_dir, "jacobian_eval_history.json")
        with open(summary_path, "w", encoding="utf-8") as fp:
            json.dump(jacobian_eval_history, fp, indent=2)

        steps = np.array([entry["step"] for entry in jacobian_eval_history])
        all_time = np.array(jacobian_eval_history[0]["time"])
        cmap = plt.colormaps.get_cmap("viridis")
        norm = plt.Normalize(vmin=steps.min(), vmax=steps.max())

        fig, ax = plt.subplots(figsize=(10, 5))
        for entry in jacobian_eval_history:
            color = cmap(norm(entry["step"]))
            ax.plot(
                all_time,
                entry["mean"],
                color=color,
                alpha=0.9,
                label=f"step {entry['step']}",
            )
        ax.set_title("Jacobian Frobenius Norm Trajectories")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Frobenius norm (mean)")
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Training step")
        if len(jacobian_eval_history) <= 10:
            ax.legend(loc="upper right", fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()

        summary_figure_path = os.path.join(jacobian_plot_dir, "jacobian_summary.png")
        fig.savefig(summary_figure_path)
        try:
            wandb.run.summary["jacobian_summary_plot"] = wandb.Image(
                fig, caption="Jacobian Frobenius trajectories"
            )
        except Exception:
            logging.exception("Failed to log jacobian summary to wandb.")
        plt.close(fig)

    for step_index in range(1, train_cfg.steps + 1):
        input_sequence, target_sequence, attention_mask = dataset()

        model_params, optimizer_state, training_metrics = train_step(
            model_params,
            optimizer_state,
            input_sequence,
            target_sequence,
            attention_mask,
        )

        if step_index % train_cfg.log_every == 0:
            train_stats = {
                metric_name: float(metric_value)
                for metric_name, metric_value in training_metrics.items()
            }
            wandb.log({"step": step_index, **train_stats})
            feature_stats = maybe_log_features(
                step_index, model_params, input_sequence, attention_mask
            )
            if feature_stats is not None:
                last_feature_metrics = feature_stats

            elapsed = time.time() - start_time
            speed = elapsed / step_index if step_index > 0 else float("inf")
            eta = (
                (train_cfg.steps - step_index) * speed
                if step_index > 0
                else float("inf")
            )
            print(
                f"\nStep {step_index}/{train_cfg.steps}  ({format_hms(elapsed)} elapsed, {format_hms(eta)} ETA)"
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
            if last_generalization_metrics:
                print(
                    "  gen  ┆ "
                    + " │ ".join(
                        f"{name}: {value:.4f}"
                        for name, value in last_generalization_metrics.items()
                    )
                )
            print("-" * 80)
        else:
            maybe_log_features(step_index, model_params, input_sequence, attention_mask)

        if step_index % train_cfg.eval_every == 0:
            aggregated_eval_metrics = {"mse": 0.0, "mae": 0.0, "nll": 0.0}
            prediction_batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
            jacobian_batch: Tuple[jnp.ndarray, jnp.ndarray] | None = None

            for _ in range(train_cfg.eval_steps):
                eval_inputs, eval_targets, eval_mask = dataset()
                if prediction_batch is None:
                    prediction_batch = (eval_inputs, eval_targets, eval_mask)
                if jacobian_batch is None:
                    jacobian_batch = (eval_inputs, eval_mask)
                eval_metrics = eval_step(
                    model_params, eval_inputs, eval_targets, eval_mask
                )
                for key in aggregated_eval_metrics:
                    aggregated_eval_metrics[key] += (
                        float(eval_metrics[key]) / train_cfg.eval_steps
                    )

            wandb.log(
                {
                    "step": step_index,
                    "eval/mse": aggregated_eval_metrics["mse"],
                    "eval/mae": aggregated_eval_metrics["mae"],
                    "eval/nll": aggregated_eval_metrics["nll"],
                }
            )

            metrics_for_checkpoint = {
                train_cfg.ckpt_metric: aggregated_eval_metrics[train_cfg.ckpt_metric]
            }
            maybe_save_best(
                step_index, metrics_for_checkpoint, model_params, optimizer_state
            )
            if not train_cfg.sweep_run:
                save_weight_checkpoint(
                    model,
                    model_params,
                    train_cfg,
                    dataset_name,
                    architecture_name,
                    step_index,
                    lr=optimizer_cfg.lr,
                    weight_decay=optimizer_cfg.weight_decay,
                )
            last_eval_metrics = {
                "eval/mse": aggregated_eval_metrics["mse"],
                "eval/mae": aggregated_eval_metrics["mae"],
                "eval/nll": aggregated_eval_metrics["nll"],
            }
            generalization_metrics: Dict[str, float] = {}
            generalization_metrics.update(
                run_length_generalization(step_index, model_params)
            )
            generalization_metrics.update(
                run_noise_generalization(step_index, model_params)
            )
            if generalization_metrics:
                last_generalization_metrics = generalization_metrics
            if jacobian_batch is not None and not train_cfg.sweep_run:
                log_jacobian_figure(step_index, model_params, *jacobian_batch)

        if (
            step_index % train_cfg.ckpt_every == 0
            and not train_cfg.disable_checkpointing
            and not train_cfg.sweep_run
            and checkpoint_manager is not None
        ):
            checkpoint_manager.save(
                step_index,
                args=ocp.args.PyTreeSave(
                    {"params": model_params, "opt_state": optimizer_state}
                ),
            )

    if not train_cfg.sweep_run:
        final_inputs, final_targets, final_mask = dataset()
        log_prediction_figure(
            train_cfg.steps, model_params, final_inputs, final_targets, final_mask
        )
        if supports_feature_logging:
            log_jacobian_figure(train_cfg.steps, model_params, final_inputs, final_mask)
        log_final_jacobian_summary()

    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="kbit_flipflop")
def main(cfg: DictConfig) -> None:
    train_cfg = TrainLoopConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    optimizer_cfg = OptimizerConfig(
        **OmegaConf.to_container(cfg.optimizer, resolve=True)
    )
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    if not isinstance(task_cfg, dict):
        raise ValueError("Task config must be a mapping.")
    train_kbit(task_cfg, cfg.model, train_cfg, optimizer_cfg)


if __name__ == "__main__":
    main()
