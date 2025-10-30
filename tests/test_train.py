import jax
from src.models.base import ModelConfig
from src.models.rnn import ElmanRNN
from src.data.dgm_dataset import DGMConfig
from src.train.train import TrainConfig, train


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
	train_cfg = TrainConfig(steps=2, log_every=1, eval_every=2, ckpt_every=1000)
	model = ElmanRNN(model_cfg)
	# Smoke test; ensure no exceptions
	train(model, model_cfg, data_cfg, train_cfg)
