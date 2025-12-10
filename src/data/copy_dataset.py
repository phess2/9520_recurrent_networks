import jax
import jax.numpy as jnp


class CopyDataset:
    def __init__(
        self,
        min_lag: int,
        max_lag: int,
        batch_size: int,
        num_classes: int = 10,
        seq_length=10,
    ):
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.key = jax.random.PRNGKey(0)
        self.seq_length = seq_length

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
        # seq_length = jax.random.randint(subkey1,shape=(),minval=10,maxval=50)
        # Actual sequence length for this batch
        # actual_length = lag + 2 * self.seq_length
        # Maximum possible length (for padding)
        max_length = self.max_lag + 2 * self.seq_length

        # initial 10-symbol sequence [batch, 10]
        init_seq = jax.random.randint(
            subkey2,
            shape=(self.batch_size, self.seq_length),
            minval=0,
            maxval=8,
        )

        # Create inputs padded to max_length
        inputs = jnp.full((self.batch_size, max_length), fill_value=8, dtype=jnp.int32)
        inputs = inputs.at[:, : self.seq_length].set(init_seq)

        # set delimeter
        delimiter_pos = self.seq_length + lag - 1
        inputs = inputs.at[:, delimiter_pos].set(9)

        # Create targets padded to max_length
        targets = jnp.full((self.batch_size, max_length), fill_value=9, dtype=jnp.int32)
        copy_start = lag + self.seq_length
        targets = targets.at[:, copy_start : copy_start + self.seq_length].set(init_seq)

        # Mask: 1 only where target token is non-null (i.e., requires prediction)
        mask = (targets != 9).astype(jnp.float32)

        return inputs, targets, mask
