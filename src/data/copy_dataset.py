import jax
import jax.numpy as jnp

class CopyDataset:
    def __init__(self, seq_length_min:int,seq_length_max:int, lag_min: int, lag_max: int, batch_size: int, num_classes: int = 10):
        self.seq_length_min = seq_length_min
        self.seq_length_max = seq_length_max
        self.lag_min = lag_min
        self.lag_max = lag_max
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.key = jax.random.PRNGKey(0)

    def __call__(self):
        return self.make_copy_batch()

    def make_copy_batch(self):
        assert self.num_classes == 10

        key, subkey = jax.random.split(self.key)

        # Sample a single lag shared by the whole batch
        lag_key, seq_key = jax.random.split(subkey)
        lag = jax.random.randint(
            lag_key,
            shape=(),
            minval=self.lag_min,
            maxval=self.lag_max + 1,
        )  # scalar, in [lag_min, lag_max] inclusive

        seq_length = jax.random.randint(
            lag_key,
            shape=(),
            minval=self.seq_length_min,
            maxval=self.seq_length_max + 1,
        )  # scalar, in [lag_min, lag_max] inclusive

        # initial 10-symbol sequence [batch, 10]
        init_seq = jax.random.randint(
            seq_key,
            shape=(self.batch_size, seq_length),
            minval=0,
            maxval=8,
        )

        # Compute length for batch: 10 + lag + 10
        length = int(lag) + 2*seq_length

        # Allocate buffers
        inputs = jnp.full((self.batch_size, length), fill_value=8, dtype=jnp.int32)
        targets = jnp.full((self.batch_size, length), fill_value=9, dtype=jnp.int32)

        # Process each sample in batch
        def make_sample(i, carry):
            inp, tgt = carry
            seq = init_seq[i]
            # Write sequence to input
            inp = inp.at[i, :seq_length].set(seq)
            # Set delimiter token at position 10+lag-1
            delim_pos = seq_length + lag - 1
            inp = inp.at[i, delim_pos].set(9)
            # Write target sequence at start=10+lag, end=10+lag+10
            tgt = tgt.at[i, seq_length + lag : seq_length + lag + seq_length].set(seq)
            return (inp, tgt)

        inputs, targets = jax.lax.fori_loop(
            0, self.batch_size, make_sample, (inputs, targets)
        )

        return inputs, targets
