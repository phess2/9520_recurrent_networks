from __future__ import annotations

from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf

from ..configs.schemas import TrainLoopConfig
from ..models.base import BaseSequenceModel, ModelConfig
from ..models.lru import LinearRecurrentUnit
from ..models.rnn import ElmanRNN, LSTM
from ..models.transformer import TransformerAdapter

MODEL_REGISTRY = {
	"elman": ElmanRNN,
	"lstm": LSTM,
	"transformer": TransformerAdapter,
	"lru": LinearRecurrentUnit,
}


def register_model(name: str, model_cls) -> None:
	MODEL_REGISTRY[name.lower()] = model_cls


def build_model(model_cfg: DictConfig, train_cfg: TrainLoopConfig, task_dims: Dict[str, Any]) -> Tuple[BaseSequenceModel, ModelConfig]:
	model_dict = OmegaConf.to_container(model_cfg, resolve=True)
	if not isinstance(model_dict, dict):
		raise ValueError("Model config must be a mapping.")

	for dim_name in ("input_dim", "output_dim"):
		if dim_name not in task_dims:
			raise ValueError(f"Task configuration must provide '{dim_name}'.")

	arch = str(model_dict.get("architecture", "elman")).lower()
	input_dim = int(task_dims["input_dim"])
	output_dim = int(task_dims["output_dim"])
	hidden_dim = int(model_dict["hidden_dim"])
	num_layers = int(model_dict.get("num_layers", 1))
	precision = str(model_dict.get("precision", train_cfg.precision))

	model_config = ModelConfig(
		input_dim=input_dim,
		output_dim=output_dim,
		hidden_dim=hidden_dim,
		num_layers=num_layers,
		precision=precision,
	)

	kwargs = model_dict.get("kwargs") or {}
	if isinstance(kwargs, DictConfig):
		kwargs = OmegaConf.to_container(kwargs, resolve=True) or {}

	model_cls = MODEL_REGISTRY.get(arch)
	if model_cls is None:
		raise ValueError(f"Unknown model architecture '{arch}'.")
	return model_cls(model_config, **kwargs), model_config

