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
from ..data.copy_dataset import CopyDataset
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


def compute_metrics(
    logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Compute negative log-likelihood (NLL) and accuracy for classification task.

    Args:
        logits: Model predictions of shape [batch, seq_len, num_classes]
        target: Target token IDs of shape [batch, seq_len]
        mask: Attention mask of shape [batch, seq_len] indicating valid positions

    Returns:
        Dictionary with 'nll' (negative log-likelihood) and 'accuracy' metrics
    """
    # Compute log probabilities for all classes
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Extract log probability of the target token at each position
    # target[..., None] adds a dimension for indexing: [batch, seq_len, 1]
    log_likelihood = jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]

    # Compute mean negative log-likelihood over all valid (masked) positions
    negative_log_likelihood = -jnp.sum(log_likelihood * mask) / jnp.sum(mask)

    # Compute accuracy: fraction of positions where predicted class matches target
    predicted_classes = jnp.argmax(logits, axis=-1)
    accuracy = jnp.sum((predicted_classes == target) * mask) / jnp.sum(mask)

    return {"nll": negative_log_likelihood, "accuracy": accuracy}


def shift_targets(token_ids: jnp.ndarray, mask: jnp.ndarray):
    """Shift targets for next-token prediction task.

    For next-token prediction, we want to predict token[t+1] given tokens[0:t+1].
    This function shifts the targets so that target[t] = token[t+1].

    Args:
        token_ids: Token IDs of shape [batch, seq_len]
        mask: Attention mask of shape [batch, seq_len]

    Returns:
        Tuple of (shifted_target_ids, shifted_mask) where:
        - shifted_target_ids[t] = token_ids[t+1] for t < seq_len-1, else 0
        - shifted_mask is similarly shifted
    """
    # Shift targets: [token_1, token_2, ..., token_n] -> [token_2, token_3, ..., 0]
    # The last position gets a zero padding since there's no next token
    shifted_target_ids = jnp.concatenate(
        [token_ids[:, 1:], jnp.zeros_like(token_ids[:, :1])], axis=1
    )

    # Shift mask accordingly: mask[t] indicates if target[t] is valid
    shifted_mask = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)

    return shifted_target_ids, shifted_mask


def train_copy(
    task_cfg: Dict[str, Any],
    model_cfg: DictConfig,
    train_cfg: TrainLoopConfig,
    optimizer_cfg: OptimizerConfig,
) -> None:
    """Train an RNN on the copy task.

    The copy task requires the model to remember an input sequence and reproduce
    it after a delay period. This tests the model's ability to maintain information
    in its hidden state over time.
    """
    run_identifier, hydra_run_dir = initialize_wandb_run(
        train_cfg, task_cfg, model_cfg, optimizer_cfg
    )
    # Create dataset generator for the copy task
    dataset = CopyDataset(
        min_lag=int(task_cfg["min_lag"]),
        max_lag=int(task_cfg["max_lag"]),
        batch_size=int(task_cfg["batch_size"]),
        num_classes=int(task_cfg["num_classes"]),
        seq_length=int(task_cfg.get("seq_length", 10)),
    )

    vocab_size = int(task_cfg["num_classes"])
    num_classes = vocab_size
    input_dim = int(task_cfg.get("input_dim", vocab_size))
    output_dim = int(task_cfg.get("output_dim", vocab_size))
    task_dims = {"input_dim": input_dim, "output_dim": output_dim}

    model, model_config = build_model(model_cfg, train_cfg, task_dims)
    validate_precisions(model_config, train_cfg)

    # Build optimizer with learning rate schedule
    optimizer = build_optimizer(optimizer_cfg, train_cfg.steps)

    # Initialize model parameters and optimizer state
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
    start_time = time.time()

    dataset_name = "copy"
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

    def embed_inputs(token_ids: jnp.ndarray) -> jnp.ndarray:
        embedded = jax.nn.one_hot(token_ids, input_dim, dtype=jnp.float32)
        return maybe_cast_precision(embedded, train_cfg.precision)

    @jax.jit
    def train_step(model_params, optimizer_state, inputs, targets, mask):
        """Single training step: forward pass, loss computation, and parameter update.

        For next-token prediction, we predict target[t] given inputs[0:t+1].
        The targets are shifted so that target[t] corresponds to the next token.
        """
        # Shift targets for next-token prediction task
        shifted_target_ids, shifted_target_mask = shift_targets(targets, mask)

        def loss_fn(current_params):
            """Compute loss and metrics for current parameters."""
            embedded_inputs = embed_inputs(inputs)

            # Forward pass: get logits for all positions
            logits = model.apply(current_params, embedded_inputs, mask)

            # Compute metrics (NLL and accuracy)
            metrics = compute_metrics(logits, shifted_target_ids, shifted_target_mask)

            # Return loss (NLL) and metrics as auxiliary output
            return metrics["nll"], metrics

        # Compute gradients and loss value simultaneously
        (negative_log_likelihood, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(model_params)

        # Update optimizer state and compute parameter updates
        parameter_updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, model_params
        )

        # Apply updates to parameters
        model_params = optax.apply_updates(model_params, parameter_updates)

        return model_params, optimizer_state, metrics

    @jax.jit
    def eval_step(model_params, inputs, targets, mask):
        """Single evaluation step: forward pass and metric computation (no gradients)."""
        # Shift targets for next-token prediction
        shifted_target_ids, shifted_target_mask = shift_targets(targets, mask)

        embedded_inputs = jax.nn.one_hot(inputs, input_dim, dtype=jnp.float32)
        embedded_inputs = embed_inputs(inputs)

        # Forward pass: get predictions
        logits = model.apply(model_params, embedded_inputs, mask)

        # Compute metrics
        metrics = compute_metrics(logits, shifted_target_ids, shifted_target_mask)

        return metrics

    # Track best metric value for checkpointing
    best_metric_value = None
    # Determine if higher is better (accuracy) or lower is better (NLL)
    higher_is_better = train_cfg.ckpt_metric.lower() == "accuracy"

    def maybe_save_best(
        step_index: int, metrics: Dict[str, Any], model_params, optimizer_state
    ):
        """Save checkpoint if current metric is the best seen so far.

        Compares current metric value to best seen value and saves if improved.
        For accuracy, higher is better; for NLL, lower is better.
        """
        current_metric_value = float(metrics[train_cfg.ckpt_metric])
        nonlocal best_metric_value

        # Check if this is an improvement
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

            # Log best metric to wandb summary
            wandb.run.summary["best_" + train_cfg.ckpt_metric] = current_metric_value

            # Save checkpoint if checkpointing is enabled
            if (
                train_cfg.save_best
                and not train_cfg.disable_checkpointing
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
        input_ids: jnp.ndarray,
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

        def process(ids, mask_arr):
            embedded = embed_inputs(ids)
            _, tensors, stats = model.analyze_batch(model_params, embedded, mask_arr)
            return tensors, stats

        for idx in range(feature_log_batches):
            if idx == 0:
                ids = input_ids
                mask_arr = mask
            else:
                ids, _, mask_arr = dataset()
            tensors, stats = process(ids, mask_arr)
            mask_arr = mask_arr.astype(jnp.float32)
            mask_total += float(jnp.sum(mask_arr))
            frob_sum += float(jnp.sum(stats.frobenius_norms * mask_arr))
            active_sum += float(jnp.sum(stats.nonlinearity_active_fraction))
            scaling_sum += float(jnp.sum(stats.nonlinearity_scaling))
            scaling_count += float(
                jnp.sum(mask_arr) * stats.nonlinearity_scaling.shape[-1]
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
        if payload_to_save is not None:
            save_path = os.path.join(feature_save_dir, f"step_{step_index:06d}.npz")
            payload_np = jax.device_get(payload_to_save)
            np.savez(save_path, **payload_np)
        return feature_metrics

    def create_prediction_figure(
        model_params,
        input_ids: jnp.ndarray,
        target_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
        example_idx: int = 0,
    ):
        embedded_inputs = embed_inputs(input_ids)
        logits = model.apply(model_params, embedded_inputs, attention_mask)
        predictions = jnp.argmax(logits, axis=-1)

        inputs_np = np.asarray(jax.device_get(input_ids))
        targets_np = np.asarray(jax.device_get(target_ids))
        preds_np = np.asarray(jax.device_get(predictions))
        mask_np = np.asarray(jax.device_get(attention_mask)).astype(bool)

        example_idx = int(np.clip(example_idx, 0, inputs_np.shape[0] - 1))
        inp = inputs_np[example_idx]
        tgt = targets_np[example_idx]
        pred = preds_np[example_idx]
        valid = mask_np[example_idx]

        fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
        axes[0].imshow(
            inp[None, :], aspect="auto", cmap="tab20", vmin=0, vmax=num_classes
        )
        axes[0].set_ylabel("input")
        axes[1].imshow(
            tgt[None, :], aspect="auto", cmap="tab20", vmin=0, vmax=num_classes
        )
        axes[1].set_ylabel("target")
        axes[2].imshow(
            pred[None, :], aspect="auto", cmap="tab20", vmin=0, vmax=num_classes
        )
        axes[2].set_ylabel("pred")
        axes[3].imshow(
            valid[None, :].astype(float), aspect="auto", cmap="gray", vmin=0, vmax=1
        )
        axes[3].set_ylabel("mask")
        for ax in axes:
            ax.set_yticks([])
            ax.set_xlim(0, len(inp))
        axes[3].set_xlabel("time step")
        plt.tight_layout()

        valid_count = np.maximum(valid.sum(), 1)
        diff = (pred != tgt) & valid
        sequence_accuracy = 1.0 - (diff.sum() / valid_count)
        return fig, float(sequence_accuracy)

    def log_prediction_figure(
        step_index: int,
        model_params,
        input_ids: jnp.ndarray,
        target_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ):
        try:
            fig, sequence_accuracy = create_prediction_figure(
                model_params, input_ids, target_ids, attention_mask
            )
        except Exception:
            logging.exception("Failed to create prediction visualization.")
            return

        caption = f"Step {step_index} - Sequence accuracy {sequence_accuracy:.3f}"
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
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        embedded = embed_inputs(input_ids)
        full_mask = jnp.ones_like(attention_mask)
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
        input_ids: jnp.ndarray,
        attention_mask: jnp.ndarray,
    ):
        if not supports_feature_logging:
            return
        try:
            time_points, mean, std = compute_jacobian_stats(
                model_params, input_ids, attention_mask
            )
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
        if not jacobian_eval_history:
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

    # Main training loop
    for step_index in range(1, train_cfg.steps + 1):
        # Get a batch of training data (now includes mask)
        input_sequence, target_sequence, attention_mask = dataset()

        # Perform one training step: forward pass, backward pass, and parameter update
        model_params, optimizer_state, training_metrics = train_step(
            model_params,
            optimizer_state,
            input_sequence,
            target_sequence,
            attention_mask,
        )

        # Log training metrics periodically
        if step_index % train_cfg.log_every == 0:
            train_stats = {
                metric_name: float(metric_value)
                for metric_name, metric_value in training_metrics.items()
            }
            wandb.log(
                {
                    "step": step_index,
                    **train_stats,
                }
            )
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
            print("-" * 80)
        else:
            maybe_log_features(step_index, model_params, input_sequence, attention_mask)

        # Run evaluation periodically
        if step_index % train_cfg.eval_every == 0:
            # Aggregate metrics over multiple evaluation batches for more stable estimates
            aggregated_eval_metrics = {"nll": 0.0, "accuracy": 0.0}
            prediction_batch: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None
            jacobian_batch: tuple[jnp.ndarray, jnp.ndarray] | None = None

            for _ in range(train_cfg.eval_steps):
                eval_inputs, eval_targets, eval_mask = dataset()
                if prediction_batch is None:
                    prediction_batch = (eval_inputs, eval_targets, eval_mask)
                if jacobian_batch is None:
                    jacobian_batch = (eval_inputs, eval_mask)
                eval_metrics = eval_step(
                    model_params, eval_inputs, eval_targets, eval_mask
                )

                # Accumulate metrics (averaging will happen by dividing by eval_steps)
                aggregated_eval_metrics["nll"] += (
                    float(eval_metrics["nll"]) / train_cfg.eval_steps
                )
                aggregated_eval_metrics["accuracy"] += (
                    float(eval_metrics["accuracy"]) / train_cfg.eval_steps
                )

            # Log evaluation metrics
            wandb.log(
                {
                    "step": step_index,
                    "eval/nll": aggregated_eval_metrics["nll"],
                    "eval/accuracy": aggregated_eval_metrics["accuracy"],
                }
            )

            # Check if this is the best model and save if so
            metrics_for_checkpoint = {
                train_cfg.ckpt_metric: aggregated_eval_metrics[train_cfg.ckpt_metric]
            }
            maybe_save_best(
                step_index, metrics_for_checkpoint, model_params, optimizer_state
            )
            save_weight_checkpoint(
                model,
                model_params,
                train_cfg,
                dataset_name,
                architecture_name,
                step_index,
            )
            last_eval_metrics = {
                "eval/nll": aggregated_eval_metrics["nll"],
                "eval/accuracy": aggregated_eval_metrics["accuracy"],
            }
            if jacobian_batch is not None and not train_cfg.sweep_run:
                log_jacobian_figure(step_index, model_params, *jacobian_batch)

        # Save periodic checkpoints (not just best model)
        if (
            step_index % train_cfg.ckpt_every == 0
            and not train_cfg.disable_checkpointing
            and checkpoint_manager is not None
        ):
            checkpoint_manager.save(
                step_index,
                args=ocp.args.PyTreeSave(
                    {"params": model_params, "opt_state": optimizer_state}
                ),
            )

    # Log final prediction visualization
    if not train_cfg.sweep_run:
        final_inputs, final_targets, final_mask = dataset()
        log_prediction_figure(
            train_cfg.steps, model_params, final_inputs, final_targets, final_mask
        )
        if supports_feature_logging:
            log_jacobian_figure(train_cfg.steps, model_params, final_inputs, final_mask)
        log_final_jacobian_summary()

    def maybe_log_features(
        step_index: int,
        model_params,
        input_ids: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        if feature_log_every <= 0 or (step_index % feature_log_every) != 0:
            return
        if not supports_feature_logging:
            return

        mask_total = 0.0
        frob_sum = 0.0
        active_sum = 0.0
        scaling_sum = 0.0
        scaling_count = 0.0
        lambda_max_value = None
        sigma_max_value = None
        payload_to_save = None

        def process(ids, mask_arr):
            embedded = embed_inputs(ids)
            _, tensors, stats = model.analyze_batch(model_params, embedded, mask_arr)
            return tensors, stats

        for idx in range(feature_log_batches):
            if idx == 0:
                ids = input_ids
                mask_arr = mask
            else:
                ids, _, mask_arr = dataset()
            tensors, stats = process(ids, mask_arr)
            mask_arr = mask_arr.astype(jnp.float32)
            mask_total += float(jnp.sum(mask_arr))
            frob_sum += float(jnp.sum(stats.frobenius_norms * mask_arr))
            active_sum += float(jnp.sum(stats.nonlinearity_active_fraction))
            scaling_sum += float(jnp.sum(stats.nonlinearity_scaling))
            scaling_count += float(
                jnp.sum(mask_arr) * stats.nonlinearity_scaling.shape[-1]
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

        if mask_total <= 0:
            return
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
        if payload_to_save is not None:
            save_path = os.path.join(feature_save_dir, f"step_{step_index:06d}.npz")
            payload_np = jax.device_get(payload_to_save)
            np.savez(save_path, **payload_np)

    # Finalize wandb run
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="copy")
def main(cfg: DictConfig) -> None:
    train_cfg = TrainLoopConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    optimizer_cfg = OptimizerConfig(
        **OmegaConf.to_container(cfg.optimizer, resolve=True)
    )
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    if not isinstance(task_cfg, dict):
        raise ValueError("Task config must be a mapping.")
    train_copy(task_cfg, cfg.model, train_cfg, optimizer_cfg)


if __name__ == "__main__":
    main()
