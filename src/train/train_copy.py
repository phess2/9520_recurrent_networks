from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from typing import Any, Dict

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import DictConfig, OmegaConf
import re
import numpy as np
from src.configs.schemas import OptimizerConfig, TrainLoopConfig
from src.models.base import ModelConfig
from src.data.copy_dataset import CopyDataset
from src.train.model_factory import build_model


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


# Suppress Orbax checkpointing warnings about blocking main thread
# These warnings are informational and don't indicate an error
class OrbaxWarningFilter(logging.Filter):
    """Filter to suppress Orbax _SignalingThread.join() warnings."""

    def filter(self, record):
        # Suppress warnings about _SignalingThread.join() blocking the main thread
        if "_SignalingThread.join()" in record.getMessage():
            return False
        return True


# Apply filter to absl logger (where Orbax warnings come from)
absl_logger = logging.getLogger("absl")
absl_logger.addFilter(OrbaxWarningFilter())


def build_optimizer(
    config: OptimizerConfig, total_steps: int
) -> optax.GradientTransformation:
    """Build optimizer with learning rate schedule."""
    if config.scheduler == "linear":
        learning_rate_schedule = optax.linear_schedule(
            init_value=0.0, end_value=config.lr, transition_steps=config.warmup_steps
        )
    elif config.scheduler == "cosine":
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.lr,
            warmup_steps=config.warmup_steps,
            decay_steps=max(total_steps - config.warmup_steps, 1),
            end_value=0.0,
        )
    elif config.scheduler == "none":
        learning_rate_schedule = config.lr
    else:
        raise ValueError(f"Unknown scheduler {config.scheduler}")

    name = config.name.lower()
    if name == "adamw":
        return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
    elif name == "sgd":
        return optax.sgd(learning_rate_schedule, momentum=0.9, nesterov=True)
    elif name == "muon":
        return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {config.name}")


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


def maybe_cast_precision(array: jnp.ndarray, precision: str) -> jnp.ndarray:
    """Cast array to specified precision if needed.

    Args:
        array: Input array
        precision: Precision string (e.g., "bfloat16")

    Returns:
        Array cast to specified precision, or original array if precision not recognized
    """
    if precision == "bfloat16":
        return array.astype(jnp.bfloat16)
    return array


def _format_hms(seconds: float) -> str:
    if not jnp.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _shift_mask(mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)


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
    # Initialize wandb for experiment tracking
    if not train_cfg.entity:
        raise ValueError("train.entity must be set to the target W&B team.")
    if train_cfg.wandb_api_key:
        wandb.login(key=train_cfg.wandb_api_key, relogin=True)
    wandb.init(
        project=train_cfg.project,
        name=train_cfg.run_name,
        entity=train_cfg.entity,
        config={
            "task": task_cfg,
            "model": OmegaConf.to_container(model_cfg, resolve=True),
            "train": asdict(train_cfg),
            "optimizer": asdict(optimizer_cfg),
        },
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
    input_dim = int(task_cfg.get("input_dim", vocab_size))
    output_dim = int(task_cfg.get("output_dim", vocab_size))
    task_dims = {"input_dim": input_dim, "output_dim": output_dim}

    model, model_config = build_model(model_cfg, train_cfg, task_dims)
    _validate_precisions(model_config, train_cfg)

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

    # Set up checkpoint manager if orbax is available
    checkpoint_manager = None
    # Build checkpoint directory: checkpoints/{dataset_name}/{architecture_name}
    dataset_name = "copy"
    architecture_name = re.sub(
        r"(?<!^)(?=[A-Z])", "_", model.__class__.__name__
    ).lower()
    checkpoint_directory = os.path.abspath(
        os.path.join("checkpoints", dataset_name, architecture_name)
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_directory, ocp.PyTreeCheckpointer()
    )

    def embed_inputs(token_ids: jnp.ndarray) -> jnp.ndarray:
        embedded = jax.nn.one_hot(token_ids, input_dim, dtype=jnp.float32)
        return maybe_cast_precision(embedded, train_cfg.precision)

    @jax.jit
    def train_step(model_params, optimizer_state, inputs, targets, mask, focus_mask):
        """Single training step: forward pass, loss computation, and parameter update.

        For next-token prediction, we predict target[t] given inputs[0:t+1].
        The targets are shifted so that target[t] corresponds to the next token.
        """
        # Shift targets for next-token prediction task
        shifted_target_ids, shifted_target_mask = shift_targets(targets, mask)
        shifted_focus_mask = _shift_mask(focus_mask)

        def loss_fn(current_params):
            """Compute loss and metrics for current parameters."""
            embedded_inputs = embed_inputs(inputs)

            # Forward pass: get logits for all positions
            logits = model.apply(current_params, embedded_inputs, mask)

            # Compute metrics (NLL and accuracy)
            effective_mask = shifted_target_mask * shifted_focus_mask
            metrics = compute_metrics(logits, shifted_target_ids, effective_mask)

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
    def eval_step(model_params, inputs, targets, mask, focus_mask):
        """Single evaluation step: forward pass and metric computation (no gradients)."""
        # Shift targets for next-token prediction
        shifted_target_ids, shifted_target_mask = shift_targets(targets, mask)
        shifted_focus_mask = _shift_mask(focus_mask)

        embedded_inputs = jax.nn.one_hot(inputs, input_dim, dtype=jnp.float32)
        embedded_inputs = embed_inputs(inputs)

        # Forward pass: get predictions
        logits = model.apply(model_params, embedded_inputs, mask)

        # Compute metrics
        effective_mask = shifted_target_mask * shifted_focus_mask
        metrics = compute_metrics(logits, shifted_target_ids, effective_mask)

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
            if train_cfg.save_best and checkpoint_manager is not None:
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
        if feature_log_every <= 0 or (step_index % feature_log_every) != 0:
            return None
        if not supports_feature_logging:
            return None

        mask_total = 0.0
        frob_sum = 0.0
        active_sum = 0.0
        scaling_sum = 0.0
        scaling_count = 0.0
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
                payload_to_save = payload

        if mask_total <= 0:
            return None
        eps = 1e-6
        feature_metrics = {
            "features/frobenius_mean": frob_sum / (mask_total + eps),
            "features/nonlinearity_active_fraction": active_sum / (mask_total + eps),
            "features/nonlinearity_scale_mean": scaling_sum / (scaling_count + eps),
        }
        wandb.log({"step": step_index, **feature_metrics})
        if payload_to_save is not None:
            save_path = os.path.join(feature_save_dir, f"step_{step_index:06d}.npz")
            payload_np = jax.device_get(payload_to_save)
            np.savez(save_path, **payload_np)
        return feature_metrics

    # Main training loop
    for step_index in range(1, train_cfg.steps + 1):
        # Get a batch of training data (now includes mask)
        input_sequence, target_sequence, attention_mask = dataset()
        focus_mask = attention_mask

        # Perform one training step: forward pass, backward pass, and parameter update
        model_params, optimizer_state, training_metrics = train_step(
            model_params,
            optimizer_state,
            input_sequence,
            target_sequence,
            attention_mask,
            focus_mask,
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
                f"\nStep {step_index}/{train_cfg.steps}  ({_format_hms(elapsed)} elapsed, {_format_hms(eta)} ETA)"
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

            for _ in range(train_cfg.eval_steps):
                eval_inputs, eval_targets, eval_mask = dataset()
                eval_metrics = eval_step(
                    model_params, eval_inputs, eval_targets, eval_mask, eval_mask
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
            last_eval_metrics = {
                "eval/nll": aggregated_eval_metrics["nll"],
                "eval/accuracy": aggregated_eval_metrics["accuracy"],
            }

        # Save periodic checkpoints (not just best model)
        if step_index % train_cfg.ckpt_every == 0 and checkpoint_manager is not None:
            checkpoint_manager.save(
                step_index,
                args=ocp.args.PyTreeSave(
                    {"params": model_params, "opt_state": optimizer_state}
                ),
            )

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
            if payload_to_save is None and feature_save_dir:
                payload_to_save = {
                    "frobenius_norms": stats.frobenius_norms,
                    "active_fraction": stats.nonlinearity_active_fraction,
                    "scaling": stats.nonlinearity_scaling,
                    "pre_activations": tensors.pre_activations,
                    "hidden_states": tensors.hidden_states,
                }

        if mask_total <= 0:
            return
        eps = 1e-6
        feature_metrics = {
            "features/frobenius_mean": frob_sum / (mask_total + eps),
            "features/nonlinearity_active_fraction": active_sum / (mask_total + eps),
            "features/nonlinearity_scale_mean": scaling_sum / (scaling_count + eps),
        }
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
