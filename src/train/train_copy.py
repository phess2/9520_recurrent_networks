from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

# Handle relative imports when running as script
# Add project root to path before other imports
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Imports after sys.path modification to allow both module and script execution
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
import orbax.checkpoint as ocp  # noqa: E402

import wandb  # noqa: E402

# Try relative imports first (when run as module), fall back to absolute (when run as script)
try:
    from ..data.copy_dataset import CopyDataset
    from ..models.base import BaseSequenceModel, ModelConfig
    from ..models.rnn import ElmanRNN, LSTM
    from ..models.lru import LinearRecurrentUnit
    from ..models.transformer import TransformerAdapter
except ImportError:
    # Fall back to absolute imports when running as script
    from src.data.copy_dataset import CopyDataset
    from src.models.base import BaseSequenceModel, ModelConfig
    from src.models.rnn import ElmanRNN, LSTM
    from src.models.lru import LinearRecurrentUnit
    from src.models.transformer import TransformerAdapter


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

# Type aliases for optimizer and scheduler names
OptimizerName = Literal["adamw", "sgd", "muon"]
SchedulerName = Literal["linear", "cosine"]


@dataclass
class CopyTrainConfig:
    """Configuration for training an RNN on the copy task.

    The copy task tests a model's ability to remember and reproduce a sequence
    after a delay period (lag). The lag is sampled uniformly from [min_lag, max_lag]
    for each batch, and the copy sequence length is sampled uniformly from
    [min_copy_length, max_copy_length], providing a range of difficulty levels.
    This is a classic benchmark for recurrent networks.
    """

    # Wandb logging configuration
    project: str = "recurrent_networks_copy"
    run_name: str = ""  # Will be set to hess_copy_{model_name} if empty

    # Model configuration
    model_name: str = "elman_rnn"  # Options: "elman_rnn", "lstm", "lru", "transformer"
    model_kwargs: Dict[str, Any] = field(
        default_factory=dict
    )  # Additional kwargs for model initialization

    # Training loop configuration
    steps: int = 5000
    log_every: int = steps // 100  # Log training metrics every N steps
    eval_every: int = steps // 10  # Run evaluation every N steps
    eval_steps: int = steps // 5  # Number of batches to evaluate over

    # Checkpointing configuration
    ckpt_every: int = steps // 20  # Save checkpoint every N steps
    ckpt_metric: str = (
        "accuracy"  # Metric to track for best model ("accuracy" or "nll")
    )
    save_best: bool = True  # Whether to save best model based on ckpt_metric
    save_periodic_checkpoints: bool = False  # Whether to save periodic checkpoints (in addition to best model)

    # Model precision
    precision: str = "bfloat16"

    # Optimizer configuration
    optimizer: OptimizerName = "adamw"
    lr: float = 1e-3  # Learning rate
    weight_decay: float = 0.0
    scheduler: SchedulerName = "linear"  # Learning rate schedule type
    warmup_steps: int = 100  # Number of warmup steps for learning rate schedule

    # Copy task specific hyperparameters
    min_lag: int = 10  # Minimum delay between input sequence and target output
    max_lag: int = 100  # Maximum delay between input sequence and target output
    min_copy_length: int = 1  # Minimum length of the sequence to be copied
    max_copy_length: int = 10  # Maximum length of the sequence to be copied
    batch_size: int = 32
    num_classes: int = 10  # Vocabulary size (number of distinct tokens)
    embed_dim: int = 32  # Embedding dimension for token inputs
    hidden_dim: int = 128  # Hidden dimension of the RNN


class ModelWithEmbedding(BaseSequenceModel):
    """Wrapper that adds embedding layer for discrete token inputs.

    This class wraps any BaseSequenceModel with an embedding layer to convert discrete
    token IDs into continuous embeddings before passing them to the model.
    """

    def __init__(
        self,
        base_model: BaseSequenceModel,
        vocab_size: int,
        embed_dim: int,
        config: ModelConfig,
    ):
        super().__init__(config)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.base_model = base_model

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

        # Initialize base model parameters
        base_params = self.base_model.initialize(rnn_key)

        return {"embedding": embedding_params, "model": base_params}

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

        # Apply base model to embedded sequence
        return self.base_model.apply(params["model"], embeddings, mask)


def create_base_model(
    model_name: str, model_config: ModelConfig, model_kwargs: Dict[str, Any]
) -> BaseSequenceModel:
    """Factory function to create a base model instance.

    Args:
        model_name: Name of the model to create ("elman_rnn", "lstm", "lru", "transformer")
        model_config: ModelConfig instance with model hyperparameters
        model_kwargs: Additional keyword arguments specific to the model type

    Returns:
        BaseSequenceModel instance

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "elman_rnn":
        return ElmanRNN(model_config)
    elif model_name == "lstm":
        return LSTM(model_config)
    elif model_name == "lru":
        return LinearRecurrentUnit(model_config)
    elif model_name == "transformer":
        return TransformerAdapter(model_config, **model_kwargs)
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Supported options: 'elman_rnn', 'lstm', 'lru', 'transformer'"
        )


def build_optimizer(config: CopyTrainConfig) -> optax.GradientTransformation:
    """Build optimizer with learning rate schedule.

    Creates a learning rate schedule and wraps it with the specified optimizer.
    """
    # Build learning rate schedule based on configuration
    if config.scheduler == "linear":
        # Linear warmup from 0 to learning_rate over warmup_steps
        learning_rate_schedule = optax.linear_schedule(
            init_value=0.0, end_value=config.lr, transition_steps=config.warmup_steps
        )
    elif config.scheduler == "cosine":
        # Cosine schedule: warmup to peak, then cosine decay to zero
        learning_rate_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.lr,
            warmup_steps=config.warmup_steps,
            decay_steps=max(config.steps - config.warmup_steps, 1),
            end_value=0.0,
        )
    else:
        raise ValueError(f"Unknown scheduler {config.scheduler}")

    # Build optimizer with the learning rate schedule
    if config.optimizer == "adamw":
        return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        # SGD with momentum and Nesterov acceleration
        return optax.sgd(learning_rate_schedule, momentum=0.9, nesterov=True)
    elif config.optimizer == "muon":
        return optax.adamw(learning_rate_schedule, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {config.optimizer}")


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


def train_copy(config: CopyTrainConfig) -> None:
    """Train a model on the copy task.

    The copy task requires the model to remember an input sequence and reproduce
    it after a delay period. This tests the model's ability to maintain information
    in its hidden state over time.

    Supports multiple model types: elman_rnn, lstm, lru, and transformer.
    """
    # Set run name if not provided
    if not config.run_name:
        config.run_name = f"hess_copy_{config.model_name}"

    # Print training configuration
    print("=" * 80)
    print(f"Starting training: {config.run_name}")
    print("=" * 80)
    print(f"Model: {config.model_name}")
    print(f"Model kwargs: {config.model_kwargs}")
    print(f"Steps: {config.steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler}")
    print(f"Lag range: [{config.min_lag}, {config.max_lag}]")
    print(f"Copy length range: [{config.min_copy_length}, {config.max_copy_length}]")
    print(f"Hidden dim: {config.hidden_dim}, Embed dim: {config.embed_dim}")
    print(f"Checkpointing: save_best={config.save_best}, save_periodic={config.save_periodic_checkpoints}")
    print("=" * 80)

    # Initialize wandb for experiment tracking
    wandb.init(project=config.project, name=config.run_name, config=config.__dict__)

    # Create dataset generator for the copy task
    dataset = CopyDataset(
        min_lag=config.min_lag,
        max_lag=config.max_lag,
        min_copy_length=config.min_copy_length,
        max_copy_length=config.max_copy_length,
        batch_size=config.batch_size,
        num_classes=config.num_classes,
    )

    # Create model configuration
    # Note: input_dim is set to embed_dim, but the actual input will be token IDs
    # which get embedded before being passed to the model
    model_config = ModelConfig(
        input_dim=config.embed_dim,
        output_dim=config.num_classes,
        hidden_dim=config.hidden_dim,
        precision=config.precision,
    )

    # Create base model using factory function
    base_model = create_base_model(
        config.model_name, model_config, config.model_kwargs
    )

    # Wrap base model with embedding layer
    model = ModelWithEmbedding(
        base_model=base_model,
        vocab_size=config.num_classes,
        embed_dim=config.embed_dim,
        config=model_config,
    )

    # Build optimizer with learning rate schedule
    optimizer = build_optimizer(config)

    # Initialize model parameters and optimizer state
    random_key = jax.random.PRNGKey(0)
    model_params = model.initialize(random_key)
    optimizer_state = optimizer.init(model_params)

    # Set up checkpoint manager if orbax is available
    checkpoint_manager = None
    # Build checkpoint directory: checkpoints/{dataset_name}/{architecture_name}
    dataset_name = "copy"
    architecture_name = f"{config.model_name}_with_embedding"
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
    higher_is_better = config.ckpt_metric.lower() == "accuracy"

    def maybe_save_best(
        step_index: int, metrics: Dict[str, Any], model_params, optimizer_state
    ):
        """Save checkpoint if current metric is the best seen so far.

        Compares current metric value to best seen value and saves if improved.
        For accuracy, higher is better; for NLL, lower is better.
        """
        current_metric_value = float(metrics[config.ckpt_metric])
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
            wandb.run.summary["best_" + config.ckpt_metric] = current_metric_value

            # Save checkpoint if checkpointing is enabled
            if config.save_best and checkpoint_manager is not None:
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
    for step_index in range(1, config.steps + 1):
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
        if step_index % config.log_every == 0:
            nll = float(training_metrics["nll"])
            acc = float(training_metrics["accuracy"])
            progress_pct = 100.0 * step_index / config.steps
            print(
                f"Step {step_index:5d}/{config.steps} ({progress_pct:5.1f}%) | "
                f"Train NLL: {nll:.4f} | Train Acc: {acc:.4f}"
            )
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
        if step_index % config.eval_every == 0:
            # Aggregate metrics over multiple evaluation batches for more stable estimates
            aggregated_eval_metrics = {"nll": 0.0, "accuracy": 0.0}

            for _ in range(config.eval_steps):
                eval_inputs, eval_targets, eval_mask = dataset()
                eval_metrics = eval_step(
                    model_params, eval_inputs, eval_targets, eval_mask
                )

                # Accumulate metrics (averaging will happen by dividing by eval_steps)
                aggregated_eval_metrics["nll"] += (
                    float(eval_metrics["nll"]) / config.eval_steps
                )
                aggregated_eval_metrics["accuracy"] += (
                    float(eval_metrics["accuracy"]) / config.eval_steps
                )

            # Log evaluation metrics
            eval_nll = aggregated_eval_metrics["nll"]
            eval_acc = aggregated_eval_metrics["accuracy"]
            progress_pct = 100.0 * step_index / config.steps
            print(
                f"Step {step_index:5d}/{config.steps} ({progress_pct:5.1f}%) | "
                f"Eval NLL: {eval_nll:.4f} | Eval Acc: {eval_acc:.4f}"
            )
            wandb.log(
                {
                    "step": step_index,
                    "eval/nll": eval_nll,
                    "eval/accuracy": eval_acc,
                }
            )

            # Check if this is the best model and save if so
            metrics_for_checkpoint = {
                config.ckpt_metric: aggregated_eval_metrics[config.ckpt_metric]
            }
            maybe_save_best(
                step_index, metrics_for_checkpoint, model_params, optimizer_state
            )
            if best_metric_value is not None:
                print(f"  -> Best {config.ckpt_metric}: {best_metric_value:.4f}")

        # Save periodic checkpoints if enabled
        if (
            config.save_periodic_checkpoints
            and step_index % config.ckpt_every == 0
            and checkpoint_manager is not None
        ):
            checkpoint_manager.save(
                step_index,
                args=ocp.args.PyTreeSave(
                    {"params": model_params, "opt_state": optimizer_state}
                ),
            )
            print(f"Step {step_index:5d}/{config.steps} | Saved periodic checkpoint")

    # Finalize wandb run
    print("=" * 80)
    print(f"Training completed: {config.run_name}")
    if best_metric_value is not None:
        print(f"Best {config.ckpt_metric}: {best_metric_value:.4f}")
    print("=" * 80)
    wandb.finish()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with defaults matching CopyTrainConfig."""
    parser = argparse.ArgumentParser(
        description="Train a model on the copy task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Wandb logging configuration
    parser.add_argument(
        "--project",
        type=str,
        default="recurrent_networks_copy",
        help="Wandb project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Wandb run name (will be set to hess_copy_{model_name} if empty)",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="elman_rnn",
        choices=["elman_rnn", "lstm", "lru", "transformer"],
        help="Model type to train",
    )
    parser.add_argument(
        "--model_kwargs",
        type=str,
        default="{}",
        help="Additional model kwargs as JSON string (e.g., '{\"num_heads\": 4}')",
    )

    # Training loop configuration
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=None,
        help="Log training metrics every N steps (default: steps // 100)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=None,
        help="Run evaluation every N steps (default: steps // 10)",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Number of batches to evaluate over (default: steps // 5)",
    )

    # Checkpointing configuration
    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=None,
        help="Save checkpoint every N steps (default: steps // 20)",
    )
    parser.add_argument(
        "--ckpt_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "nll"],
        help="Metric to track for best model",
    )
    parser.add_argument(
        "--no_save_best",
        dest="save_best",
        action="store_false",
        default=True,
        help="Disable saving best model (default: save_best=True)",
    )
    parser.add_argument(
        "--save_periodic_checkpoints",
        action="store_true",
        default=False,
        help="Whether to save periodic checkpoints (in addition to best model)",
    )

    # Model precision
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        help="Model precision",
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "muon"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay coefficient",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Learning rate schedule type",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate schedule",
    )

    # Copy task specific hyperparameters
    parser.add_argument(
        "--min_lag",
        type=int,
        default=10,
        help="Minimum delay between input sequence and target output",
    )
    parser.add_argument(
        "--max_lag",
        type=int,
        default=100,
        help="Maximum delay between input sequence and target output",
    )
    parser.add_argument(
        "--min_copy_length",
        type=int,
        default=1,
        help="Minimum length of the sequence to be copied",
    )
    parser.add_argument(
        "--max_copy_length",
        type=int,
        default=10,
        help="Maximum length of the sequence to be copied",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help="Vocabulary size (number of distinct tokens)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=32,
        help="Embedding dimension for token inputs",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of the RNN",
    )

    args = parser.parse_args()

    # Parse model_kwargs JSON string
    try:
        args.model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in --model_kwargs: {e}")

    # Set computed defaults if not provided
    if args.log_every is None:
        args.log_every = args.steps // 100
    if args.eval_every is None:
        args.eval_every = args.steps // 10
    if args.eval_steps is None:
        args.eval_steps = args.steps // 5
    if args.ckpt_every is None:
        args.ckpt_every = args.steps // 20

    return args


def args_to_config(args: argparse.Namespace) -> CopyTrainConfig:
    """Convert parsed arguments to CopyTrainConfig."""
    return CopyTrainConfig(
        project=args.project,
        run_name=args.run_name,
        model_name=args.model_name,
        model_kwargs=args.model_kwargs,
        steps=args.steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        ckpt_every=args.ckpt_every,
        ckpt_metric=args.ckpt_metric,
        save_best=args.save_best,
        save_periodic_checkpoints=args.save_periodic_checkpoints,
        precision=args.precision,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        warmup_steps=args.warmup_steps,
        min_lag=args.min_lag,
        max_lag=args.max_lag,
        min_copy_length=args.min_copy_length,
        max_copy_length=args.max_copy_length,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    # Parse command line arguments and start training
    args = parse_args()
    training_config = args_to_config(args)
    train_copy(training_config)
