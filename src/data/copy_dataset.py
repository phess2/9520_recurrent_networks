import jax
import jax.numpy as jnp

class CopyDataset:
    def __init__(self, lag: int, batch_size: int, num_classes: int = 10):
        self.lag = lag
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.key = jax.random.PRNGKey(0)

    def __call__(self):
        return self.make_copy_batch()

    def make_copy_batch(self):
        assert self.num_classes == 10
        length = self.lag + 20
        key, subkey = jax.random.split(self.key)

        # initial 10-symbol sequence [batch, 10]
        init_seq = jax.random.randint(
            subkey,
            shape=(self.batch_size, 10),
            minval=0,
            maxval=8,
        )

        inputs = jnp.full((self.batch_size, length), fill_value=8, dtype=jnp.int32)
        inputs = inputs.at[:, :10].set(init_seq)

        # set delimeter
        delimiter_pos = 10 + self.lag - 1
        inputs = inputs.at[:, delimiter_pos].set(9)
        targets = jnp.full((self.batch_size, length), fill_value=9, dtype=jnp.int32)
        copy_start = self.lag + 10
        targets = targets.at[:, copy_start:copy_start + 10].set(init_seq)

        return inputs, targets
