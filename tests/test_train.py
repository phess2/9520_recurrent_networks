import sys
from pathlib import Path

import jax

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.configs.schemas import OptimizerConfig, TrainLoopConfig
from src.models.base import ModelConfig
from src.models.rnn import ElmanRNN
from src.data.dgm_dataset import DGMConfig
from src.train.train import train


def test_train_smoke():
	model_cfg = ModelConfig(input_dim=6, output_dim=6, hidden_dim=8)
	data_cfg = DGMConfig(
		dims=4,
		obs_dims=6,
		lags=[1],
		discrete_latent=False,
		mean_seq_len=8,
		std_seq_len=0,
		batch_size=2,
		burn_in=2,
	)
	train_cfg = TrainLoopConfig(
		project="test",
		run_name="smoke",
		steps=2,
		log_every=1,
		eval_every=2,
		eval_steps=1,
		ckpt_every=1000,
		ckpt_metric="nll",
		save_best=False,
		precision="float32",
		seed=0,
	)
	optimizer_cfg = OptimizerConfig(
		name="adamw",
		lr=1e-3,
		weight_decay=0.0,
		scheduler="linear",
		warmup_steps=1,
		use_modula_optim=False,
	)
	model = ElmanRNN(model_cfg)
	# Smoke test; ensure no exceptions
	train(model, model_cfg, data_cfg, train_cfg, optimizer_cfg)
