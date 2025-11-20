from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import DictConfig, OmegaConf

from ..configs.schemas import OptimizerConfig, TrainLoopConfig
from ..models.base import ModelConfig


class OrbaxWarningFilter(logging.Filter):
    """Filter to suppress Orbax _SignalingThread.join() warnings."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "_SignalingThread.join()" not in record.getMessage()


_ORBAX_LOGGING_CONFIGURED = False


def configure_orbax_logging() -> None:
    """Ensure Orbax/absl logging is filtered once per process."""
    global _ORBAX_LOGGING_CONFIGURED
    if _ORBAX_LOGGING_CONFIGURED:
        return
    absl_logger = logging.getLogger("absl")
    absl_logger.addFilter(OrbaxWarningFilter())
    absl_logger.setLevel(logging.WARNING)
    logging.getLogger("absl.logging").setLevel(logging.WARNING)
    _ORBAX_LOGGING_CONFIGURED = True


def validate_precisions(model_cfg: ModelConfig, train_cfg: TrainLoopConfig) -> None:
    if model_cfg.precision != train_cfg.precision:
        raise ValueError(
            f"model.precision ({model_cfg.precision}) must match train.precision ({train_cfg.precision})."
        )
    param_dtype = model_cfg.param_dtype or model_cfg.precision
    if param_dtype != train_cfg.precision:
        raise ValueError(
            f"model.param_dtype ({param_dtype}) must match train.precision ({train_cfg.precision})."
        )


def build_optimizer(
    config: OptimizerConfig, total_steps: int
) -> optax.GradientTransformation:
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
    if name == "sgd":
        return optax.sgd(learning_rate_schedule, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer {config.name}")


def maybe_cast_precision(array, precision: str):
    if precision == "bfloat16":
        return array.astype(jnp.bfloat16)
    return array


def format_hms(seconds: float) -> str:
    if not jnp.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def initialize_wandb_run(
    train_cfg: TrainLoopConfig,
    task_cfg: Dict[str, Any],
    model_cfg: DictConfig,
    optimizer_cfg: OptimizerConfig,
) -> Tuple[str, str]:
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
    run_identifier = train_cfg.run_name or f"run_{int(time.time())}"
    if wandb.run is not None:
        run_identifier = wandb.run.name or wandb.run.id or run_identifier
    run_identifier = re.sub(r"[^a-zA-Z0-9._-]", "_", run_identifier)

    hydra_run_dir = os.environ.get("HYDRA_RUN_DIR")
    if hydra_run_dir:
        hydra_run_dir = os.path.abspath(hydra_run_dir)
        os.makedirs(hydra_run_dir, exist_ok=True)
    else:
        hydra_run_dir = os.getcwd()
    logging.info("Hydra run directory resolved to %s", hydra_run_dir)
    return run_identifier, hydra_run_dir


def create_checkpoint_manager(
    train_cfg: TrainLoopConfig, dataset_name: str, architecture_name: str
) -> Optional[ocp.CheckpointManager]:
    if train_cfg.disable_checkpointing:
        return None
    checkpoint_directory = os.path.abspath(
        os.path.join("checkpoints", dataset_name, architecture_name)
    )
    return ocp.CheckpointManager(checkpoint_directory, ocp.PyTreeCheckpointer())


def ensure_prediction_artifacts(
    train_cfg: TrainLoopConfig,
    hydra_run_dir: str,
    dataset_name: str,
    architecture_name: str,
    run_identifier: str,
) -> Tuple[Optional[str], Optional[str]]:
    if train_cfg.sweep_run:
        return None, None
    prediction_plot_dir = os.path.join(
        hydra_run_dir,
        "prediction_plots",
        dataset_name,
        architecture_name,
        run_identifier,
    )
    jacobian_plot_dir = os.path.join(
        hydra_run_dir,
        "jacobian_plots",
        dataset_name,
        architecture_name,
        run_identifier,
    )
    os.makedirs(prediction_plot_dir, exist_ok=True)
    os.makedirs(jacobian_plot_dir, exist_ok=True)
    return prediction_plot_dir, jacobian_plot_dir

