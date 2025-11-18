import jax
import jax.numpy as jnp

class CopyDataset:
    def __init__(self, min_lag: int, max_lag: int, batch_size: int, num_classes: int = 10):
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.key = jax.random.PRNGKey(0)

    def __call__(self):
        return self.make_copy_batch()

    def make_copy_batch(self):
        assert self.num_classes == 10
        # Sample a random lag for this batch
        key, subkey1, subkey2 = jax.random.split(self.key, 3)
        self.key = key
        lag = jax.random.randint(
            subkey1,
            shape=(),
            minval=self.min_lag,
            maxval=self.max_lag + 1,  # +1 because maxval is exclusive
        )
        
        # Actual sequence length for this batch
        actual_length = lag + 20
        # Maximum possible length (for padding)
        max_length = self.max_lag + 20

        # initial 10-symbol sequence [batch, 10]
        init_seq = jax.random.randint(
            subkey2,
            shape=(self.batch_size, 10),
            minval=0,
            maxval=8,
        )

        # Create inputs padded to max_length
        inputs = jnp.full((self.batch_size, max_length), fill_value=8, dtype=jnp.int32)
        inputs = inputs.at[:, :10].set(init_seq)

        # set delimeter
        delimiter_pos = 10 + lag - 1
        inputs = inputs.at[:, delimiter_pos].set(9)
        
        # Create targets padded to max_length
        targets = jnp.full((self.batch_size, max_length), fill_value=9, dtype=jnp.int32)
        copy_start = lag + 10
        targets = targets.at[:, copy_start:copy_start + 10].set(init_seq)

        # Create attention mask: 1 for valid positions, 0 for padding
        mask = jnp.zeros((self.batch_size, max_length), dtype=jnp.float32)
        mask = mask.at[:, :actual_length].set(1.0)

        return inputs, targets, mask