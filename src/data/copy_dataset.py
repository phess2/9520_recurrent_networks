import jax
import jax.numpy as jnp


class CopyDataset:
    def __init__(
        self,
        min_lag: int,
        max_lag: int,
        batch_size: int,
        num_classes: int = 10,
        min_copy_length: int = 1,
        max_copy_length: int = 10,
    ):
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.min_copy_length = min_copy_length
        self.max_copy_length = max_copy_length
        self.key = jax.random.PRNGKey(0)

    def __call__(self):
        return self.make_copy_batch()

    def make_copy_batch(self):
        assert self.num_classes == 10
        # Sample a random lag and copy length for this batch
        key, subkey1, subkey2, subkey3 = jax.random.split(self.key, 4)
        self.key = key
        lag = jax.random.randint(
            subkey1,
            shape=(),
            minval=self.min_lag,
            maxval=self.max_lag + 1,  # +1 because maxval is exclusive
        )
        copy_length = jax.random.randint(
            subkey2,
            shape=(),
            minval=self.min_copy_length,
            maxval=self.max_copy_length + 1,  # +1 because maxval is exclusive
        )

        # Actual sequence length for this batch
        # Structure: [copy_length symbols] + [lag-1 fill tokens] + [delimiter] + [lag fill tokens] + [copy_length copied symbols]
        actual_length = copy_length + lag + copy_length
        # Maximum possible length (for padding)
        max_length = self.max_copy_length + self.max_lag + self.max_copy_length

        # initial sequence with variable length [batch, copy_length]
        init_seq = jax.random.randint(
            subkey3,
            shape=(self.batch_size, copy_length),
            minval=0,
            maxval=8,
        )

        # Create inputs padded to max_length
        inputs = jnp.full((self.batch_size, max_length), fill_value=8, dtype=jnp.int32)
        inputs = inputs.at[:, :copy_length].set(init_seq)

        # set delimeter
        delimiter_pos = copy_length + lag - 1
        inputs = inputs.at[:, delimiter_pos].set(9)

        # Create targets padded to max_length
        targets = jnp.full((self.batch_size, max_length), fill_value=9, dtype=jnp.int32)
        copy_start = lag + copy_length
        targets = targets.at[:, copy_start : copy_start + copy_length].set(init_seq)

        # Create attention mask: 1 for valid positions, 0 for padding
        mask = jnp.zeros((self.batch_size, max_length), dtype=jnp.float32)
        mask = mask.at[:, :actual_length].set(1.0)

        return inputs, targets, mask
