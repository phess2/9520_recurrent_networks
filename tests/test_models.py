import jax
import jax.numpy as jnp

from src.models.base import ModelConfig
from src.models.rnn import ElmanRNN, LSTM
from src.models.lru import LinearRecurrentUnit


def test_elman_forward():
	cfg = ModelConfig(input_dim=8, output_dim=8, hidden_dim=16)
	model = ElmanRNN(cfg)
	params = model.initialize(jax.random.PRNGKey(0))
	x = jnp.zeros((2, 5, 8))
	mask = jnp.ones((2, 5))
	out = model.apply(params, x, mask)
	assert out.shape == (2, 5, 8)


def test_lstm_forward():
	cfg = ModelConfig(input_dim=8, output_dim=8, hidden_dim=16)
	model = LSTM(cfg)
	params = model.initialize(jax.random.PRNGKey(0))
	x = jnp.zeros((2, 5, 8))
	mask = jnp.ones((2, 5))
	out = model.apply(params, x, mask)
	assert out.shape == (2, 5, 8)


def test_lru_forward():
	cfg = ModelConfig(input_dim=8, output_dim=8, hidden_dim=16)
	model = LinearRecurrentUnit(cfg)
	params = model.initialize(jax.random.PRNGKey(0))
	x = jnp.zeros((2, 5, 8))
	mask = jnp.ones((2, 5))
	out = model.apply(params, x, mask)
	assert out.shape == (2, 5, 8)
