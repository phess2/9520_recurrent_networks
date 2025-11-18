from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any, Dict

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import DictConfig, OmegaConf
from src.configs.schemas import OptimizerConfig, TrainLoopConfig
from src.data.copy_dataset import CopyDataset
from src.models.base import BaseSequenceModel, ModelConfig
from src.models.rnn import ElmanRNN


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

class RNNWithEmbedding(BaseSequenceModel):
    """RNN wrapper that adds embedding layer for discrete token inputs.

    This class wraps an ElmanRNN with an embedding layer to convert discrete
    token IDs into continuous embeddings before passing them to the RNN.
    """

    def __init__(self, config: ModelConfig, vocab_size: int, embed_dim: int):
        # Create a model config for the RNN that uses embedding dimension as input
        rnn_config = ModelConfig(
            input_dim=embed_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            precision=config.precision,
        )
        super().__init__(rnn_config)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rnn = ElmanRNN(rnn_config)

    def initialize(self, key: jax.Array) -> Any:
        """Initialize model parameters: embedding matrix and RNN weights."""
        # Split random key for independent initialization of embedding and RNN
        embedding_key, rnn_key = jax.random.split(key, 2)

        # Initialize embedding matrix: [vocab_size, embed_dim]
        # Each row represents the embedding vector for a token ID
        # These are learnable parameters that will be updated during training via backpropagation
        # Gradients flow through the embedding lookup operation in the forward pass
        embedding_params = jax.nn.initializers.normal(stddev=0.02)(
            embedding_key, (self.vocab_size, self.embed_dim), dtype=jnp.float32
        )

        # Initialize RNN parameters
        rnn_params = self.rnn.initialize(rnn_key)

        return {"embedding": embedding_params, "rnn": rnn_params}

    def apply(
        self, params: Any, token_ids: jnp.ndarray, mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass: embed tokens and apply RNN.

        Args:
            params: Model parameters containing 'embedding' and 'rnn' keys
            token_ids: Token IDs of shape [batch, seq_len]
            mask: Attention mask of shape [batch, seq_len]

        Returns:
            RNN outputs of shape [batch, seq_len, output_dim]
        """
        # Look up embeddings for each token ID: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embeddings = params["embedding"][token_ids]

        # Apply RNN to embedded sequence
        return self.rnn.apply(params["rnn"], embeddings, mask)


def build_optimizer(config: OptimizerConfig, total_steps: int) -> optax.GradientTransformation:
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


def train_copy(
    task_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    train_cfg: TrainLoopConfig,
    optimizer_cfg: OptimizerConfig,
) -> None:
    """Train an RNN on the copy task.

    The copy task requires the model to remember an input sequence and reproduce
    it after a delay period. This tests the model's ability to maintain information
    in its hidden state over time.
    """
    # Initialize wandb for experiment tracking
    wandb.init(
        project=train_cfg.project,
        name=train_cfg.run_name,
        config={
            "task": task_cfg,
            "model": model_cfg,
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
    )

    # Create model configuration
    # Note: input_dim is set to embed_dim, but the actual input will be token IDs
    # which get embedded before being passed to the RNN
    embed_dim = int(model_cfg["embed_dim"])
    hidden_dim = int(model_cfg["hidden_dim"])
    num_layers = int(model_cfg.get("num_layers", 1))
    model_config = ModelConfig(
        input_dim=embed_dim,
        output_dim=int(task_cfg["num_classes"]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        precision=train_cfg.precision,
    )

    # Initialize model with embedding layer and RNN
    model = RNNWithEmbedding(
        model_config,
        vocab_size=int(task_cfg["num_classes"]),
        embed_dim=embed_dim,
    )

    # Build optimizer with learning rate schedule
    optimizer = build_optimizer(optimizer_cfg, train_cfg.steps)

    # Initialize model parameters and optimizer state
    random_key = jax.random.PRNGKey(train_cfg.seed)
    model_params = model.initialize(random_key)
    optimizer_state = optimizer.init(model_params)

    # Set up checkpoint manager if orbax is available
    checkpoint_manager = None
    # Build checkpoint directory: checkpoints/{dataset_name}/{architecture_name}
    dataset_name = "copy"
    architecture_name = "rnn_with_embedding"  # RNNWithEmbedding wraps ElmanRNN
    checkpoint_directory = os.path.abspath(
        os.path.join("checkpoints", dataset_name, architecture_name)
    )
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_directory, ocp.PyTreeCheckpointer()
    )

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
            # Forward pass: get logits for all positions
            logits = model.apply(current_params, inputs, mask)

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

        # Forward pass: get predictions
        logits = model.apply(model_params, inputs, mask)

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
            wandb.log(
                {
                    "step": step_index,
                    **{
                        metric_name: float(metric_value)
                        for metric_name, metric_value in training_metrics.items()
                    },
                }
            )

        # Run evaluation periodically
        if step_index % train_cfg.eval_every == 0:
            # Aggregate metrics over multiple evaluation batches for more stable estimates
            aggregated_eval_metrics = {"nll": 0.0, "accuracy": 0.0}

            for _ in range(train_cfg.eval_steps):
                eval_inputs, eval_targets, eval_mask = dataset()
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

        # Save periodic checkpoints (not just best model)
        if step_index % train_cfg.ckpt_every == 0 and checkpoint_manager is not None:
            checkpoint_manager.save(
                step_index,
                args=ocp.args.PyTreeSave(
                    {"params": model_params, "opt_state": optimizer_state}
                ),
            )

    # Finalize wandb run
    wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="copy")
def main(cfg: DictConfig) -> None:
    train_cfg = TrainLoopConfig(**OmegaConf.to_container(cfg.train, resolve=True))
    optimizer_cfg = OptimizerConfig(**OmegaConf.to_container(cfg.optimizer, resolve=True))
    task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(task_cfg, dict) or not isinstance(model_cfg, dict):
        raise ValueError("Task and model configs must be mappings.")
    train_copy(task_cfg, model_cfg, train_cfg, optimizer_cfg)


if __name__ == "__main__":
    main()
