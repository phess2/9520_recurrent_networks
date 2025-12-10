from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict

import jax
import jax.numpy as jnp


@dataclass
class DGMConfig:
    """Configuration for multivariate DGM generator."""

    dims: int
    obs_dims: int
    lags: List[int]
    discrete_latent: bool
    num_categories: int = 8
    mean_seq_len: int = 256
    std_seq_len: int = 0
    batch_size: int = 32
    burn_in: int = 32
    seed: int = 0
    cov_scale: float = 0.2
    input_dim: int | None = None
    output_dim: int | None = None

    def __post_init__(self) -> None:
        if self.input_dim is None:
            self.input_dim = self.obs_dims
        if self.output_dim is None:
            self.output_dim = self.obs_dims


class DGMDataset:
    """Multivariate DGM generator yielding variable-length batches as JAX arrays.

    Generates latent states with specified lag dependencies and distinct
    emission models (Bernoulli/noisy one-hot for discrete latents; Gaussian for continuous).
    """

    def __init__(self, config: DGMConfig):
        self.config = config
        self.key = jax.random.PRNGKey(config.seed)

        # Random default parameterization placeholders. In future, allow overrides via config.
        self._init_random_params()

    def _init_random_params(self) -> None:
        c = self.config
        if c.discrete_latent:
            # Per-lag transition logits: [num_lags, num_categories, num_categories]
            if len(c.lags) == 0:
                self.trans_logits = jnp.zeros((0, c.num_categories, c.num_categories))
            else:
                self.key, k = jax.random.split(self.key)
                self.trans_logits = (
                    jax.random.normal(
                        k, (len(c.lags), c.num_categories, c.num_categories)
                    )
                    * 0.1
                )
            # Emission: categorical observation ids in [0, obs_dims). Optionally noisy one-hot continuous view.
        else:
            # Continuous: Transition weights for each lag: shape [len(lags), dims, dims]
            if len(c.lags) == 0:
                self.trans_weights = jnp.zeros((0, c.dims, c.dims))
            else:
                self.key, k = jax.random.split(self.key)
                w = jax.random.normal(k, (len(c.lags), c.dims, c.dims)) * (
                    c.cov_scale / jnp.sqrt(c.dims)
                )
                self.trans_weights = w
        # Emission mapping (for continuous), and biases
        self.key, k_e = jax.random.split(self.key)
        self.emission = jax.random.normal(k_e, (c.obs_dims, max(1, c.dims))) * (
            1.0 / jnp.sqrt(max(1, c.dims))
        )
        self.key, k_b1, k_b2 = jax.random.split(self.key, 3)
        self.trans_bias = jax.random.normal(k_b1, (max(1, c.dims),)) * 0.01
        self.emission_bias = jax.random.normal(k_b2, (c.obs_dims,)) * 0.01
        # Noise scales
        self.transition_noise = 0.05
        self.emission_noise = 0.1

    def _sample_lengths(self, key: jax.Array, batch_size: int) -> jnp.ndarray:
        c = self.config
        if c.std_seq_len <= 0:
            return jnp.full((batch_size,), c.mean_seq_len, dtype=jnp.int32)
        self.key, k = jax.random.split(key)
        lengths = jnp.clip(
            jax.random.normal(k, (batch_size,)) * c.std_seq_len + c.mean_seq_len,
            a_min=1,
            a_max=None,
        ).astype(jnp.int32)
        return lengths

    def _transition_discrete(self, prev_states: List[int]) -> int:
        c = self.config
        if len(c.lags) == 0:
            # IID categories
            self.key, k = jax.random.split(self.key)
            probs = jnp.ones((c.num_categories,)) / c.num_categories
            return int(jax.random.categorical(k, jnp.log(probs)))
        # Sum logits over lags conditioned on previous categories
        logits = jnp.zeros((c.num_categories,))
        for i, _lag in enumerate(c.lags):
            prev_cat = int(prev_states[i]) if i < len(prev_states) else 0
            logits = logits + self.trans_logits[i, prev_cat]
        self.key, k = jax.random.split(self.key)
        return int(jax.random.categorical(k, logits))

    def _transition_continuous(self, prev_states: List[jnp.ndarray]) -> jnp.ndarray:
        c = self.config
        # x_t = sum_lag W_l x_{t-lag} + b + eps
        mean = self.trans_bias
        for i, _lag in enumerate(c.lags):
            mean = mean + self.trans_weights[i] @ prev_states[i]
        self.key, k = jax.random.split(self.key)
        return mean + self.transition_noise * jax.random.normal(k, (c.dims,))

    def _emit(self, latent) -> Tuple[jnp.ndarray, Optional[int]]:
        c = self.config
        if c.discrete_latent:
            cat = int(latent)
            # Discrete observation ids via a simple noisy channel mapping categories to obs classes
            # Here we use a random permutation/noise via linear bias; for now, map cat mod obs_dims
            obs_id = cat % c.obs_dims
            self.key, k = jax.random.split(self.key)
            # Noisy one-hot view for convenience
            one_hot = jax.nn.one_hot(obs_id, c.obs_dims, dtype=jnp.float32)
            noisy = jnp.clip(
                one_hot + 0.1 * jax.random.normal(k, (c.obs_dims,)), 0.0, 1.0
            )
            return noisy, obs_id
        else:
            # y_t = E x_t + b + eps
            self.key, k = jax.random.split(self.key)
            cont = (
                self.emission @ latent
                + self.emission_bias
                + self.emission_noise * jax.random.normal(k, (self.config.obs_dims,))
            )
            return cont, None

    def sample_batch(self) -> Dict[str, jnp.ndarray]:
        """Sample a variable-length batch and return padded arrays with masks.

        Returns a dict with keys: 'latents', 'observations', 'lengths', 'mask'.
        If discrete_latent=True, also returns 'latent_ids' and 'obs_ids'.
        """
        c = self.config
        self.key, k_len = jax.random.split(self.key)
        lengths = self._sample_lengths(k_len, c.batch_size)
        max_len = int(lengths.max()) + c.burn_in

        latents_list = []
        obs_list = []
        latent_ids_list: List[jnp.ndarray] = []
        obs_ids_list: List[jnp.ndarray] = []
        for b in range(c.batch_size):
            T = int(lengths[b]) + c.burn_in
            # Initialize history with zeros
            if c.discrete_latent:
                state_hist: List[int] = [0 for _ in c.lags]
            else:
                state_hist: List[jnp.ndarray] = [jnp.zeros((c.dims,)) for _ in c.lags]
            seq_latents = []
            seq_latent_ids = []
            seq_obs = []
            seq_obs_ids = []
            for t in range(T):
                prev = state_hist[:]
                if c.discrete_latent:
                    new_latent_id = self._transition_discrete(prev)
                    obs_vec, obs_id = self._emit(new_latent_id)
                    seq_latent_ids.append(jnp.int32(new_latent_id))
                    seq_obs_ids.append(jnp.int32(obs_id))
                    seq_latents.append(jnp.int32(new_latent_id))
                    seq_obs.append(obs_vec)
                    new_state_to_hist = new_latent_id
                else:
                    new_latent = self._transition_continuous(prev)
                    obs_vec, _ = self._emit(new_latent)
                    seq_latents.append(new_latent)
                    seq_obs.append(obs_vec)
                    new_state_to_hist = new_latent
                # update history (FIFO over specified lags)
                if len(state_hist) > 0:
                    state_hist = [new_state_to_hist] + state_hist[:-1]
            if c.discrete_latent:
                latents_arr = jnp.array(seq_latent_ids)
                obs_arr = jnp.stack(seq_obs, axis=0)
                obs_ids_arr = jnp.array(seq_obs_ids)
                latent_ids_arr = jnp.array(seq_latent_ids)
            else:
                latents_arr = jnp.stack(seq_latents, axis=0)
                obs_arr = jnp.stack(seq_obs, axis=0)
                obs_ids_arr = None
                latent_ids_arr = None
            # drop burn-in
            latents_arr = latents_arr[c.burn_in :]
            obs_arr = obs_arr[c.burn_in :]
            latents_list.append(latents_arr)
            obs_list.append(obs_arr)
            if c.discrete_latent:
                obs_ids_list.append(obs_ids_arr[c.burn_in :])
                latent_ids_list.append(latent_ids_arr[c.burn_in :])

        # Pad to common length
        final_max = int(lengths.max())
        if self.config.discrete_latent:
            lat_shape = (c.batch_size, final_max)
            lat_pad = jnp.zeros(lat_shape, dtype=jnp.int32)
            obs_ids_pad = jnp.zeros((c.batch_size, final_max), dtype=jnp.int32)
        else:
            lat_shape = (c.batch_size, final_max, c.dims)
            lat_pad = jnp.zeros(lat_shape, dtype=jnp.float32)
            obs_ids_pad = None
        obs_pad = jnp.zeros((c.batch_size, final_max, c.obs_dims), dtype=jnp.float32)
        mask = jnp.zeros((c.batch_size, final_max), dtype=jnp.float32)
        for b in range(c.batch_size):
            T = int(lengths[b])
            lb = latents_list[b]
            ob = obs_list[b]
            if self.config.discrete_latent:
                lat_pad = lat_pad.at[b, :T].set(lb[:T])
                obs_pad = obs_pad.at[b, :T, :].set(ob[:T, :])
                obs_ids_pad = obs_ids_pad.at[b, :T].set(obs_ids_list[b][:T])
            else:
                lat_pad = lat_pad.at[b, :T, :].set(lb[:T, :])
                obs_pad = obs_pad.at[b, :T, :].set(ob[:T, :])
            mask = mask.at[b, :T].set(1.0)

        out = {
            "latents": lat_pad,
            "observations": obs_pad,
            "lengths": lengths,
            "mask": mask,
        }
        if self.config.discrete_latent:
            out["latent_ids"] = lat_pad
            out["obs_ids"] = obs_ids_pad
        return out

    def iterator(self) -> Iterator[Dict[str, jnp.ndarray]]:
        while True:
            yield self.sample_batch()
