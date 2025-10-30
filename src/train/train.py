from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any

import jax
import jax.numpy as jnp
import optax
import wandb

from ..data.dgm_dataset import DGMDataset, DGMConfig
from ..models.base import BaseSequenceModel, ModelConfig

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

from ..utils.metrics import mutual_information_placeholder

OptimizerName = Literal["adamw", "sgd", "muon"]
SchedulerName = Literal["linear", "cosine"]


@dataclass
class TrainConfig:
	project: str = "recurrent_networks_dgm"
	run_name: str = "debug"
	steps: int = 1000
	log_every: int = 50
	eval_every: int = 200
	eval_steps: int = 10
	ckpt_every: int = 200
	ckpt_metric: str = "nll"  # or "accuracy"
	save_best: bool = True
	precision: str = "bfloat16"
	optimizer: OptimizerName = "adamw"
	lr: float = 3e-4
	weight_decay: float = 0.0
	scheduler: SchedulerName = "linear"
	warmup_steps: int = 100
	use_modula_optim: bool = True


def build_optimizer(cfg: TrainConfig) -> optax.GradientTransformation:
	if cfg.scheduler == "linear":
		schedule = optax.linear_schedule(init_value=0.0, end_value=cfg.lr, transition_steps=cfg.warmup_steps)
	elif cfg.scheduler == "cosine":
		schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=cfg.lr, warmup_steps=cfg.warmup_steps, decay_steps=max(cfg.steps - cfg.warmup_steps, 1), end_value=0.0)
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


def compute_metrics(pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray, discrete: bool) -> Dict[str, jnp.ndarray]:
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


def train(model: BaseSequenceModel, model_cfg: ModelConfig, data_cfg: DGMConfig, train_cfg: TrainConfig) -> None:
	wandb.init(project=train_cfg.project, name=train_cfg.run_name, config={"model": model_cfg.__dict__, "data": data_cfg.__dict__, "train": train_cfg.__dict__})

	dataset = DGMDataset(data_cfg)
	opt = build_optimizer(train_cfg)

	key = jax.random.PRNGKey(0)
	params = model.initialize(key)
	opt_state = opt.init(params)

	ckpt_mgr = None
	if _ORBAX_AVAILABLE:
		ckpt_mgr = ocp.CheckpointManager("checkpoints", ocp.PyTreeCheckpointer())

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
			logits = model.apply(p, maybe_cast_precision(obs, train_cfg.precision), mask)
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

	def maybe_save_best(step_idx: int, metrics: Dict[str, Any], params, opt_state):
		now = float(metrics[train_cfg.ckpt_metric])
		nonlocal best_metric
		improved = (best_metric is None) or ((now > best_metric) if best_is_higher else (now < best_metric))
		if improved:
			best_metric = now
			wandb.run.summary["best_" + train_cfg.ckpt_metric] = now
			if _ORBAX_AVAILABLE and train_cfg.save_best and ckpt_mgr is not None:
				ckpt_mgr.save(step_idx, args={"params": params, "opt_state": opt_state, "best_metric": best_metric})

	for step_idx in range(1, train_cfg.steps + 1):
		batch = dataset.sample_batch()
		if data_cfg.discrete_latent:
			params, opt_state, metrics = step_discrete(params, opt_state, batch)
		else:
			params, opt_state, metrics = step_continuous(params, opt_state, batch)

		# Mutual information placeholder (no internal states available yet)
		mi = mutual_information_placeholder(batch.get("latents"), batch.get("latents"), batch["mask"])  # placeholder

		if step_idx % train_cfg.log_every == 0:
			wandb.log({"step": step_idx, **{k: float(v) for k, v in metrics.items()}, "mi_placeholder": float(mi)})

		if step_idx % train_cfg.eval_every == 0:
			# Evaluate over eval_steps mini-batches
			agg = {"nll": 0.0, "accuracy": 0.0}
			for _ in range(train_cfg.eval_steps):
				b = dataset.sample_batch()
				m = eval_step_discrete(params, b) if data_cfg.discrete_latent else eval_step_continuous(params, b)
				agg["nll"] += float(m["nll"]) / train_cfg.eval_steps
				agg["accuracy"] += float(m["accuracy"]) / train_cfg.eval_steps
			wandb.log({"step": step_idx, "eval/nll": agg["nll"], "eval/accuracy": agg["accuracy"]})
			metrics_for_ckpt = {train_cfg.ckpt_metric: agg[train_cfg.ckpt_metric]}
			maybe_save_best(step_idx, metrics_for_ckpt, params, opt_state)

		if step_idx % train_cfg.ckpt_every == 0 and ckpt_mgr is not None:
			ckpt_mgr.save(step_idx, args={"params": params, "opt_state": opt_state})

	wandb.finish()
