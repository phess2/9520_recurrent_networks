from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

import jax
import jax.numpy as jnp
import optax
import wandb

from ..data.copy_dataset import CopyDataset
from ..models.base import BaseSequenceModel, ModelConfig
from ..models.rnn import ElmanRNN

try:
    import orbax.checkpoint as ocp

    _ORBAX_AVAILABLE = True
except Exception:
    _ORBAX_AVAILABLE = False

# Try to import Modula optimizers if present
_MODULA_MUON = None
_MODULA_ADAMW = None
try:
    from modula.optimize import muon as _muon  # type: ignore

    _MODULA_MUON = _muon
except Exception:
    try:
        from modula.optimizer import muon as _muon  # type: ignore

        _MODULA_MUON = _muon
    except Exception:
        _MODULA_MUON = None

try:
    from modula.optimize import adamw as _madamw  # type: ignore

    _MODULA_ADAMW = _madamw
except Exception:
    try:
        from modula.optimizer import adamw as _madamw  # type: ignore

        _MODULA_ADAMW = _madamw
    except Exception:
        _MODULA_ADAMW = None

OptimizerName = Literal["adamw", "sgd", "muon"]
SchedulerName = Literal["linear", "cosine"]


@dataclass
class CopyTrainConfig:
    project: str = "recurrent_networks_copy"
    run_name: str = "copy_task"
    steps: int = 5000
    log_every: int = 50
    eval_every: int = 200
    eval_steps: int = 10
    ckpt_every: int = 500
    ckpt_metric: str = "accuracy"  # or "nll"
    save_best: bool = True
    precision: str = "bfloat16"
    optimizer: OptimizerName = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: SchedulerName = "linear"
    warmup_steps: int = 100
    use_modula_optim: bool = True
    # Copy task specific
    lag: int = 10
    batch_size: int = 32
    num_classes: int = 10
    embed_dim: int = 64
    hidden_dim: int = 128


class RNNWithEmbedding(BaseSequenceModel):
    """RNN wrapper that adds embedding layer for discrete token inputs."""

    def __init__(self, config: ModelConfig, vocab_size: int, embed_dim: int):
        # Override input_dim to use embed_dim after embedding
        embed_config = ModelConfig(
            input_dim=embed_dim,
            output_dim=config.output_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            precision=config.precision,
        )
        super().__init__(embed_config)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.rnn = ElmanRNN(embed_config)

    def initialize(self, key: jax.Array) -> Any:
        k1, k2 = jax.random.split(key, 2)
        # Initialize embedding matrix: [vocab_size, embed_dim]
        embed_params = jax.nn.initializers.normal(stddev=0.02)(
            k1, (self.vocab_size, self.embed_dim), dtype=jnp.float32
        )
        rnn_params = self.rnn.initialize(k2)
        return {"embedding": embed_params, "rnn": rnn_params}

    def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # x is token IDs: [batch, seq_len]
        # Embed tokens: [batch, seq_len, embed_dim]
        embeddings = params["embedding"][x]  # [batch, seq_len, embed_dim]
        # Apply RNN
        return self.rnn.apply(params["rnn"], embeddings, mask)


def build_optimizer(cfg: CopyTrainConfig) -> optax.GradientTransformation:
    if cfg.scheduler == "linear":
        schedule = optax.linear_schedule(
            init_value=0.0, end_value=cfg.lr, transition_steps=cfg.warmup_steps
        )
    elif cfg.scheduler == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.lr,
            warmup_steps=cfg.warmup_steps,
            decay_steps=max(cfg.steps - cfg.warmup_steps, 1),
            end_value=0.0,
        )
    else:
        raise ValueError(f"Unknown scheduler {cfg.scheduler}")

    if cfg.optimizer == "adamw":
        if cfg.use_modula_optim and _MODULA_ADAMW is not None:
            try:
                return _MODULA_ADAMW(schedule, weight_decay=cfg.weight_decay)  # type: ignore
            except Exception:
                pass
        return optax.adamw(schedule, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return optax.sgd(schedule, momentum=0.9, nesterov=True)
    elif cfg.optimizer == "muon":
        if cfg.use_modula_optim and _MODULA_MUON is not None:
            try:
                return _MODULA_MUON(schedule, weight_decay=cfg.weight_decay)  # type: ignore
            except Exception:
                pass
        return optax.adamw(schedule, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def compute_metrics(
    logits: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """Compute NLL and accuracy for classification task."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ll = jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]
    nll = -jnp.sum(ll * mask) / jnp.sum(mask)
    acc = jnp.sum((jnp.argmax(logits, axis=-1) == target) * mask) / jnp.sum(mask)
    return {"nll": nll, "accuracy": acc}


def shift_targets(token_ids: jnp.ndarray, mask: jnp.ndarray):
    """Shift targets to predict next token."""
    ids_next = jnp.concatenate(
        [token_ids[:, 1:], jnp.zeros_like(token_ids[:, :1])], axis=1
    )
    mask_next = jnp.concatenate([mask[:, 1:], jnp.zeros_like(mask[:, :1])], axis=1)
    return ids_next, mask_next


def maybe_cast_precision(x: jnp.ndarray, precision: str) -> jnp.ndarray:
    if precision == "bfloat16":
        return x.astype(jnp.bfloat16)
    return x


def train_copy(cfg: CopyTrainConfig) -> None:
    """Train an RNN on the copy task."""
    wandb.init(project=cfg.project, name=cfg.run_name, config=cfg.__dict__)

    dataset = CopyDataset(
        lag=cfg.lag, batch_size=cfg.batch_size, num_classes=cfg.num_classes
    )

    # Create model with embeddings
    model_config = ModelConfig(
        input_dim=cfg.embed_dim,  # Will be overridden by embedding
        output_dim=cfg.num_classes,
        hidden_dim=cfg.hidden_dim,
        precision=cfg.precision,
    )
    model = RNNWithEmbedding(
        model_config, vocab_size=cfg.num_classes, embed_dim=cfg.embed_dim
    )

    opt = build_optimizer(cfg)

    key = jax.random.PRNGKey(0)
    params = model.initialize(key)
    opt_state = opt.init(params)

    ckpt_mgr = None
    if _ORBAX_AVAILABLE:
        ckpt_mgr = ocp.CheckpointManager("checkpoints", ocp.PyTreeCheckpointer())

    @jax.jit
    def train_step(params, opt_state, inputs, targets, mask):
        # For next-token prediction: predict targets[t] given inputs[0:t+1]
        # Shift targets so we predict the next token
        target_ids, mask_t = shift_targets(targets, mask)

        def loss_fn(p):
            logits = model.apply(p, inputs, mask)
            metrics = compute_metrics(logits, target_ids, mask_t)
            return metrics["nll"], metrics

        (nll, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def eval_step(params, inputs, targets, mask):
        # Shift targets for next-token prediction
        target_ids, mask_t = shift_targets(targets, mask)
        logits = model.apply(params, inputs, mask)
        metrics = compute_metrics(logits, target_ids, mask_t)
        return metrics

    best_metric = None
    best_is_higher = cfg.ckpt_metric.lower() == "accuracy"

    def maybe_save_best(step_idx: int, metrics: Dict[str, Any], params, opt_state):
        now = float(metrics[cfg.ckpt_metric])
        nonlocal best_metric
        improved = (best_metric is None) or (
            (now > best_metric) if best_is_higher else (now < best_metric)
        )
        if improved:
            best_metric = now
            wandb.run.summary["best_" + cfg.ckpt_metric] = now
            if _ORBAX_AVAILABLE and cfg.save_best and ckpt_mgr is not None:
                ckpt_mgr.save(
                    step_idx,
                    args={
                        "params": params,
                        "opt_state": opt_state,
                        "best_metric": best_metric,
                    },
                )

    for step_idx in range(1, cfg.steps + 1):
        inputs, targets = dataset()
        # Create mask (all positions are valid for copy task)
        mask = jnp.ones((cfg.batch_size, inputs.shape[1]), dtype=jnp.float32)

        params, opt_state, metrics = train_step(
            params, opt_state, inputs, targets, mask
        )

        if step_idx % cfg.log_every == 0:
            wandb.log({"step": step_idx, **{k: float(v) for k, v in metrics.items()}})

        if step_idx % cfg.eval_every == 0:
            # Evaluate over eval_steps mini-batches
            agg = {"nll": 0.0, "accuracy": 0.0}
            for _ in range(cfg.eval_steps):
                inp, tgt = dataset()
                m = eval_step(params, inp, tgt, mask)
                agg["nll"] += float(m["nll"]) / cfg.eval_steps
                agg["accuracy"] += float(m["accuracy"]) / cfg.eval_steps
            wandb.log(
                {
                    "step": step_idx,
                    "eval/nll": agg["nll"],
                    "eval/accuracy": agg["accuracy"],
                }
            )
            metrics_for_ckpt = {cfg.ckpt_metric: agg[cfg.ckpt_metric]}
            maybe_save_best(step_idx, metrics_for_ckpt, params, opt_state)

        if step_idx % cfg.ckpt_every == 0 and ckpt_mgr is not None:
            ckpt_mgr.save(step_idx, args={"params": params, "opt_state": opt_state})

    wandb.finish()


if __name__ == "__main__":
    cfg = CopyTrainConfig()
    train_copy(cfg)
