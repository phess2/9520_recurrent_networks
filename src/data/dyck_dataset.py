import numpy as np
import jax.numpy as jnp


class DyckDataset:
    """
    Streaming generator for the Dyck language task originally used in the
    Transformer Formal Languages repo.

    Each sample is a Dyck word drawn from a probabilistic context-free grammar
    controlled by parameters (p_val, q_val). For every time-step we return a
    multi-hot target vector that marks all valid open tokens plus the single
    closing token that would correctly complete the current stack (matching the
    behaviour of `DyckLanguage.lineToTensorOutput`).

    Inputs are padded to `upper_window` with a dedicated pad token. A binary mask
    indicates which positions correspond to real tokens.
    """

    def __init__(
        self,
        batch_size: int,
        num_pairs: int = 2,
        lower_window: int = 4,
        upper_window: int = 64,
        p_val: float = 0.5,
        q_val: float = 0.3,
        min_depth: int = 0,
        max_depth: int = -1,
        seed: int = 0,
    ):
        if num_pairs <= 0:
            raise ValueError("num_pairs must be > 0")
        if upper_window <= 0:
            raise ValueError("upper_window must be > 0")
        if lower_window <= 0 or lower_window > upper_window:
            raise ValueError("lower_window must be in (0, upper_window]")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.batch_size = batch_size
        self.num_pairs = num_pairs
        self.lower_window = lower_window
        self.upper_window = upper_window
        self.p_val = float(p_val)
        self.q_val = float(q_val)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.vocab_size = 2 * num_pairs
        self.pad_token = self.vocab_size
        self.open_tokens = [2 * i for i in range(num_pairs)]
        self.close_tokens = [2 * i + 1 for i in range(num_pairs)]
        self._token_to_pair = {}
        for pair_idx in range(num_pairs):
            self._token_to_pair[self.open_tokens[pair_idx]] = pair_idx
            self._token_to_pair[self.close_tokens[pair_idx]] = pair_idx
        self.rng = np.random.default_rng(seed)

    def __call__(self):
        return self.make_batch()

    def make_batch(self):
        sequences = [self._sample_sequence() for _ in range(self.batch_size)]
        inputs = np.full(
            (self.batch_size, self.upper_window), fill_value=self.pad_token, dtype=np.int32
        )
        targets = np.zeros(
            (self.batch_size, self.upper_window, self.vocab_size), dtype=np.float32
        )
        mask = np.zeros((self.batch_size, self.upper_window), dtype=np.float32)

        for idx, seq in enumerate(sequences):
            length = len(seq)
            inputs[idx, :length] = seq
            mask[idx, :length] = 1.0
            targets[idx, :length, :] = self._sequence_targets(seq)

        return (
            jnp.asarray(inputs, dtype=jnp.int32),
            jnp.asarray(targets, dtype=jnp.float32),
            jnp.asarray(mask, dtype=jnp.float32),
        )

    # -------------------------------------------------------------------------
    # Sampling logic
    # -------------------------------------------------------------------------

    def _sample_sequence(self):
        """Resample until we obtain a sequence satisfying length/depth bounds."""
        max_tries = 10000
        for _ in range(max_tries):
            candidate = self._generate_recursive(0)
            if not candidate:
                continue
            length = len(candidate)
            if length < self.lower_window or length > self.upper_window:
                continue
            depth = self._max_depth(candidate)
            if depth < self.min_depth:
                continue
            if self.max_depth != -1 and depth > self.max_depth:
                continue
            return candidate
        raise RuntimeError(
            "Failed to sample a Dyck sequence satisfying constraints "
            f"(len in [{self.lower_window}, {self.upper_window}], "
            f"depth in [{self.min_depth}, {self.max_depth if self.max_depth != -1 else 'inf'}])"
        )

    def _generate_recursive(self, current_size):
        """Probabilistic CFG sampler mirroring DyckLanguage.generate."""
        if current_size >= self.upper_window:
            return []

        prob = self.rng.uniform()
        if prob < self.p_val:
            pair_idx = self.rng.integers(0, self.num_pairs)
            inner = self._generate_recursive(current_size + 2)
            return (
                [self.open_tokens[pair_idx]] + inner + [self.close_tokens[pair_idx]]
                if len(inner) + 2 <= self.upper_window
                else []
            )
        if prob < self.p_val + self.q_val:
            left = self._generate_recursive(current_size)
            right = self._generate_recursive(current_size)
            return (left + right) if len(left) + len(right) <= self.upper_window else []
        return []

    def _max_depth(self, seq):
        """Compute the maximum total stack depth across the sequence."""
        depth_per_pair = [0] * self.num_pairs
        max_total = 0
        for token in seq:
            pair_idx = self._token_to_pair[token]
            if token in self.open_tokens:
                depth_per_pair[pair_idx] += 1
            else:
                depth_per_pair[pair_idx] -= 1
            total_depth = sum(depth_per_pair)
            if total_depth < 0:
                return -1  # malformed (shouldn't happen)
            if total_depth > max_total:
                max_total = total_depth
        return max_total

    # -------------------------------------------------------------------------
    # Target construction
    # -------------------------------------------------------------------------

    def _sequence_targets(self, seq):
        """Replicates DyckLanguage.lineToTensorOutput, but returns numpy array."""
        targets = np.zeros((len(seq), self.vocab_size), dtype=np.float32)
        stack = []
        for t, token in enumerate(seq):
            # Update stack to reflect state AFTER consuming token.
            if token in self.open_tokens:
                pair_idx = self._token_to_pair[token]
                stack.append(self.close_tokens[pair_idx])
            else:
                if stack:
                    stack.pop()

            # Open tokens are always valid next steps.
            targets[t, self.open_tokens] = 1.0

            # Mark the currently expected closing token, if any.
            if stack:
                targets[t, stack[-1]] = 1.0

        return targets

