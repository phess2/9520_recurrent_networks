import jax
import jax.numpy as jnp

from src.data.dgm_dataset import DGMDataset, DGMConfig

def test_dgm_dataset_continuous_batch():
	cfg = DGMConfig(
		dims=4,
		obs_dims=6,
		lags=[1, 3],
		discrete_latent=False,
		mean_seq_len=16,
		std_seq_len=0,
		batch_size=4,
		burn_in=4,
	)
	ds = DGMDataset(cfg)
	batch = ds.sample_batch()
	assert batch["observations"].shape == (4, 16, 6)
	assert batch["mask"].shape == (4, 16)
