from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptimizerName = Literal["adamw", "sgd", "muon"]
SchedulerName = Literal["linear", "cosine", "none"]


@dataclass
class OptimizerConfig:
	name: OptimizerName
	lr: float
	weight_decay: float
	scheduler: SchedulerName
	warmup_steps: int
	use_modula_optim: bool = True


@dataclass
class TrainLoopConfig:
	project: str
	run_name: str
	steps: int
	log_every: int
	eval_every: int
	eval_steps: int
	ckpt_every: int
	ckpt_metric: str
	save_best: bool
	precision: str
	entity: str
	seed: int = 0
	wandb_api_key: str | None = None

