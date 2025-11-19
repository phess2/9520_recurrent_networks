import jax
import jax.numpy as jnp

from src.models.base import ModelConfig
from src.models.rnn import ElmanRNN, LSTM, UnitaryRNN
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


def test_lstm_analyze_batch():
	cfg = ModelConfig(input_dim=4, output_dim=4, hidden_dim=6)
	model = LSTM(cfg)
	params = model.initialize(jax.random.PRNGKey(1))
	x = jnp.zeros((3, 7, 4))
	mask = jnp.ones((3, 7))
	out, tensors, stats = model.analyze_batch(params, x, mask)
	assert out.shape == (3, 7, 4)
	assert tensors.candidate_pre_activations.shape == (3, 7, 6)
	assert stats.frobenius_norms.shape == (3, 7)


def test_lru_forward():
	cfg = ModelConfig(input_dim=8, output_dim=8, hidden_dim=16)
	model = LinearRecurrentUnit(cfg)
	params = model.initialize(jax.random.PRNGKey(0))
	x = jnp.zeros((2, 5, 8))
	mask = jnp.ones((2, 5))
	out = model.apply(params, x, mask)
	assert out.shape == (2, 5, 8)


def test_lru_analyze_batch():
	cfg = ModelConfig(input_dim=3, output_dim=3, hidden_dim=4)
	model = LinearRecurrentUnit(cfg)
	params = model.initialize(jax.random.PRNGKey(3))
	x = jnp.zeros((1, 6, 3))
	mask = jnp.ones((1, 6))
	out, tensors, stats = model.analyze_batch(params, x, mask)
	assert out.shape == (1, 6, 3)
	assert tensors.hidden_states.shape == (1, 6, 4)
	assert stats.frobenius_norms.shape == (1, 6)


def test_unitary_rnn_analyze_batch():
	cfg = ModelConfig(input_dim=5, output_dim=5, hidden_dim=10)
	model = UnitaryRNN(cfg)
	params = model.initialize(jax.random.PRNGKey(2))
	x = jnp.zeros((2, 4, 5))
	mask = jnp.ones((2, 4))
	out, tensors, stats = model.analyze_batch(params, x, mask)
	assert out.shape == (2, 4, 5)
	assert tensors.pre_activations.shape == (2, 4, 10)
	assert stats.nonlinearity_active_fraction.shape == (2, 4)
