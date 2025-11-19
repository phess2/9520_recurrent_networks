import jax
import jax.numpy as jnp


class KBitFlipFlopDataset:
    """K-bit flip-flop memory task dataset.
    Each bit gets random input flip-pulses (+1 or -1) at rate p,
    and the network's output must remember the last value.
    The rest of the time input is zero + noise.
    """

    def __init__(
        self,
        k: int,
        batch_size: int,
        seq_length: int,
        p: float = 0.01,
        noise_std: float = 0.01,
    ):
        self.k = k
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.p = p
        self.noise_std = noise_std
        self.key = jax.random.PRNGKey(0)

    def __call__(self):
        return self.make_kbit_flipflop_batch()

    def make_kbit_flipflop_batch(self):
        """
        Returns:
            inputs: [batch_size, seq_length, k]   # input pulses + noise
            targets: [batch_size, seq_length, k]  # last seen pulse for each bit, held constant until next
        """
        key, subkey = jax.random.split(self.key)
        # Step 1: Sample pulse occurrence for all positions [batch, seq_len, k]
        pulse_key, sign_key, noise_key = jax.random.split(subkey, 3)
        # Bernoulli for pulse presence
        pulse_mask = jax.random.bernoulli(
            pulse_key, self.p, shape=(self.batch_size, self.seq_length, self.k)
        )
        # Sign for each pulse (+1 or -1)
        signs = jax.random.choice(
            sign_key,
            jnp.array([-1.0, 1.0]),
            shape=(self.batch_size, self.seq_length, self.k),
        )
        # Pulses (set value for pulse positions, 0 elsewhere)
        pulses = signs * pulse_mask

        # Add noise (always, including pulse times)
        noise = self.noise_std * jax.random.normal(
            noise_key, (self.batch_size, self.seq_length, self.k)
        )
        inputs = pulses + noise

        # Compute targets: for each bit, last nonzero value or 0 at t=0
        # We scan along the time dimension, accumulating the last nonzero (pulse) value
        # Need to vmap over batch and bit dims
        def flipflop_scan(pulse_seq_1d):
            def update(carry, inp):
                new_carry = jnp.where(inp != 0, inp, carry)
                return new_carry, new_carry

            _, vals = jax.lax.scan(update, 0.0, pulse_seq_1d)
            return vals

        # Vectorize over batch and bit dimensions
        targets = jax.vmap(
            lambda batch_pulses: jax.vmap(flipflop_scan)(batch_pulses.T).T
        )(pulses)

        # targets shape: [batch, seq_len, k]
        return inputs, targets
