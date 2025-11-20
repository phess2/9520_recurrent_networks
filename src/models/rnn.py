from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from ..utils.jacobian_features import JacobianFeatureSummary, compute_jacobian_features
from .base import BaseSequenceModel, ModelConfig
from .nonlinearities import Nonlinearity, get_nonlinearity

Array = jnp.ndarray


def _resolve_dtype(dtype_name: str | None) -> jnp.dtype:
    """
    Converts a string dtype name to the corresponding JAX numpy dtype.
    Returns float32 by default if dtype_name is None.
    """
    if dtype_name is None:
        return jnp.float32
    dtype_name = dtype_name.lower()
    if dtype_name in ("float32", "fp32"):
        return jnp.float32
    if dtype_name in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if dtype_name in ("float16", "fp16"):
        return jnp.float16
    raise ValueError(f"Unsupported dtype '{dtype_name}'.")


def _glorot_uniform(
    key: jax.Array, in_dim: int, out_dim: int, dtype: jnp.dtype
) -> Array:
    """
    Initializes a weight matrix using Glorot (Xavier) uniform initialization.
    This helps maintain variance of activations and gradients across layers.
    """
    initialization_limit = jnp.sqrt(6.0 / (in_dim + out_dim))
    return jax.random.uniform(
        key,
        (out_dim, in_dim),
        minval=-initialization_limit,
        maxval=initialization_limit,
        dtype=dtype,
    )


def _init_linear(
    key: jax.Array,
    in_dim: int,
    out_dim: int,
    use_bias: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Dict[str, Array]:
    """
    Initializes parameters for a linear (fully-connected) layer.
    Returns a dictionary with weight matrix 'w' and optionally bias vector 'b'.
    """
    parameters: Dict[str, Array] = {
        "w": _glorot_uniform(key, in_dim, out_dim, dtype=dtype)
    }
    if use_bias:
        parameters["b"] = jnp.zeros((out_dim,), dtype=dtype)
    return parameters


def _linear_apply(input_tensor: Array, parameters: Dict[str, Array]) -> Array:
    """
    Applies a linear transformation: output = input @ weight^T + bias.
    Handles dtype casting to match the weight matrix dtype.
    """
    weight_matrix = parameters["w"]
    output = input_tensor.astype(weight_matrix.dtype) @ weight_matrix.T
    if "b" in parameters:
        output = output + parameters["b"]
    return output.astype(weight_matrix.dtype)


def _layer_norm(input_tensor: Array, epsilon: float = 1e-5) -> Array:
    """
    Applies layer normalization: (x - mean) / sqrt(variance + epsilon).
    Normalizes across the last dimension, maintaining the original dtype.
    """
    input_float32 = input_tensor.astype(jnp.float32)
    mean_value = jnp.mean(input_float32, axis=-1, keepdims=True)
    variance = jnp.mean((input_float32 - mean_value) ** 2, axis=-1, keepdims=True)
    normalized = (input_float32 - mean_value) / jnp.sqrt(variance + epsilon)
    return normalized.astype(input_tensor.dtype)


@dataclass
class RNNRuntimeTensors:
    """
    Runtime tensors captured during RNN forward pass for analysis.
    All arrays have shape [batch_size, sequence_length, hidden_dim].
    """

    pre_activations: Array  # [B, T, H]
    hidden_states: Array  # [B, T, H]
    nonlinearity_jacobian_diag: Array  # [B, T, H]


@dataclass
class LSTMRuntimeTensors:
    """
    Runtime tensors captured during LSTM forward pass for analysis.
    All arrays have shape [batch_size, sequence_length, hidden_dim].
    """

    candidate_pre_activations: Array  # [B, T, H]
    candidate_states: Array  # [B, T, H]
    input_gates: Array  # [B, T, H]
    hidden_states: Array  # [B, T, H]
    nonlinearity_jacobian_diag: Array  # [B, T, H]


class ElmanRNN(BaseSequenceModel):
    """
    Standard Elman RNN (vanilla RNN) with configurable activation function.
    Uses input-to-hidden, hidden-to-hidden, and hidden-to-output weight matrices.
    """

    def __init__(
        self,
        config: ModelConfig,
        nonlinearity: str = "relu",
        nonlinearity_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Initializes ElmanRNN with specified configuration and activation function.
        """
        super().__init__(config)
        self._nonlinearity: Nonlinearity = get_nonlinearity(
            nonlinearity, **(nonlinearity_kwargs or {})
        )
        self.param_dtype = _resolve_dtype(config.param_dtype or config.precision)
        self.use_layer_norm = config.use_layer_norm

    def initialize(self, key: jax.Array) -> Any:
        """
        Initializes all model parameters using Glorot uniform initialization.
        Creates separate random keys for each weight matrix to ensure independence.
        """
        input_to_hidden_key, hidden_to_hidden_key, hidden_to_output_key, _ = (
            jax.random.split(key, 4)
        )
        return {
            "wx": _init_linear(
                input_to_hidden_key,
                self.config.input_dim,
                self.config.hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wh": _init_linear(
                hidden_to_hidden_key,
                self.config.hidden_dim,
                self.config.hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wo": _init_linear(
                hidden_to_output_key,
                self.config.hidden_dim,
                self.config.output_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
        }

    def apply(
        self,
        params: Any,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        return_features: bool = False,
    ) -> Tuple[jnp.ndarray, RNNRuntimeTensors] | jnp.ndarray:
        """
        Forward pass through the Elman RNN.
        Processes the input sequence step-by-step, updating hidden states and producing outputs.
        """
        batch_size, _, _ = x.shape
        hidden_dim = self.config.hidden_dim
        input_sequence = x.astype(self.param_dtype)
        initial_hidden_state = jnp.zeros(
            (batch_size, hidden_dim), dtype=self.param_dtype
        )
        activation_function = self._nonlinearity.fn
        jacobian_function = self._nonlinearity.jacobian_diag

        def step(carry, current_input):
            """
            Single RNN step: combines input and previous hidden state,
            applies activation, and produces output for this timestep.
            """
            previous_hidden_state = carry
            # Combine input projection and hidden-to-hidden projection
            pre_activation = _linear_apply(current_input, params["wx"]) + _linear_apply(
                previous_hidden_state, params["wh"]
            )
            if self.use_layer_norm:
                pre_activation = _layer_norm(pre_activation)
            current_hidden_state = activation_function(pre_activation)
            output_logits = _linear_apply(current_hidden_state, params["wo"])
            jacobian_diagonal = jacobian_function(pre_activation)
            return current_hidden_state, (
                output_logits,
                pre_activation,
                current_hidden_state,
                jacobian_diagonal,
            )

        # Process sequence using scan: input is [T, B, D], output is [T, B, ...]
        _, (outputs, pre_activations, hidden_states, jacobian_diagonals) = jax.lax.scan(
            step, initial_hidden_state, input_sequence.swapaxes(0, 1)
        )
        # Swap back to [B, T, ...] format
        output = outputs.swapaxes(0, 1)
        if not return_features:
            return output
        features = RNNRuntimeTensors(
            pre_activations=pre_activations.swapaxes(0, 1),
            hidden_states=hidden_states.swapaxes(0, 1),
            nonlinearity_jacobian_diag=jacobian_diagonals.swapaxes(0, 1),
        )
        return output, features

    def analyze_batch(
        self,
        params: Any,
        x: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, RNNRuntimeTensors, JacobianFeatureSummary]:
        """
        Runs forward pass and computes Jacobian-based statistics for analysis.
        These statistics help understand gradient flow and stability properties.
        """
        outputs, runtime_tensors = self.apply(params, x, mask, return_features=True)
        hidden_to_hidden_weight = params["wh"]["w"]
        jacobian_statistics = compute_jacobian_features(
            runtime_tensors.nonlinearity_jacobian_diag, hidden_to_hidden_weight, mask
        )
        return outputs, runtime_tensors, jacobian_statistics


class LSTM(BaseSequenceModel):
    """
    Long Short-Term Memory (LSTM) network with input, forget, output gates.
    Maintains both hidden state and cell state for better long-term memory.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes LSTM with specified configuration.
        """
        super().__init__(config)
        self.hidden_dim = config.hidden_dim
        self.param_dtype = _resolve_dtype(config.param_dtype or config.precision)
        self.use_layer_norm = config.use_layer_norm

    def initialize(self, key: jax.Array) -> Any:
        """
        Initializes LSTM parameters: 4 gates (input, forget, output, candidate) each with
        input-to-hidden and hidden-to-hidden weights, plus output projection weights.
        """
        hidden_dim = self.hidden_dim
        input_dim = self.config.input_dim
        random_keys = jax.random.split(key, 9)
        return {
            "wxi": _init_linear(
                random_keys[0],
                input_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "whi": _init_linear(
                random_keys[1],
                hidden_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wxf": _init_linear(
                random_keys[2],
                input_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "whf": _init_linear(
                random_keys[3],
                hidden_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wxo": _init_linear(
                random_keys[4],
                input_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "who": _init_linear(
                random_keys[5],
                hidden_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wxc": _init_linear(
                random_keys[6],
                input_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "whc": _init_linear(
                random_keys[7],
                hidden_dim,
                hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wo": _init_linear(
                random_keys[8],
                hidden_dim,
                self.config.output_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
        }

    def apply(
        self,
        params: Any,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        return_features: bool = False,
    ) -> Tuple[jnp.ndarray, LSTMRuntimeTensors] | jnp.ndarray:
        """
        Forward pass through the LSTM.
        Implements the standard LSTM cell with input, forget, output gates and candidate state.
        """
        batch_size, _, _ = x.shape
        hidden_dim = self.hidden_dim
        input_sequence = x.astype(self.param_dtype)
        initial_hidden_state = jnp.zeros(
            (batch_size, hidden_dim), dtype=self.param_dtype
        )
        initial_cell_state = jnp.zeros((batch_size, hidden_dim), dtype=self.param_dtype)
        tanh_activation = jnp.tanh

        def step(carry, current_input):
            """
            Single LSTM step: computes gates, updates cell state, and produces hidden state.
            """
            previous_hidden_state, previous_cell_state = carry
            # Compute three gates: input, forget, and output
            input_gate = jax.nn.sigmoid(
                _linear_apply(current_input, params["wxi"])
                + _linear_apply(previous_hidden_state, params["whi"])
            )
            forget_gate = jax.nn.sigmoid(
                _linear_apply(current_input, params["wxf"])
                + _linear_apply(previous_hidden_state, params["whf"])
            )
            output_gate = jax.nn.sigmoid(
                _linear_apply(current_input, params["wxo"])
                + _linear_apply(previous_hidden_state, params["who"])
            )
            # Compute candidate state (what to potentially add to cell state)
            candidate_pre_activation = _linear_apply(
                current_input, params["wxc"]
            ) + _linear_apply(previous_hidden_state, params["whc"])
            if self.use_layer_norm:
                candidate_pre_activation = _layer_norm(candidate_pre_activation)
            candidate_state = tanh_activation(candidate_pre_activation)
            # Update cell state: forget old info, add new candidate
            current_cell_state = (
                forget_gate * previous_cell_state + input_gate * candidate_state
            )
            # Compute hidden state from cell state
            current_hidden_state = output_gate * tanh_activation(current_cell_state)
            output_logits = _linear_apply(current_hidden_state, params["wo"])
            # Jacobian diagonal for candidate state (derivative of tanh)
            jacobian_diagonal = 1.0 - candidate_state**2
            return (current_hidden_state, current_cell_state), (
                output_logits,
                candidate_pre_activation,
                candidate_state,
                input_gate,
                jacobian_diagonal,
                current_hidden_state,
            )

        # Process sequence: input is [T, B, D], output is [T, B, ...]
        (
            _,
            (
                outputs,
                candidate_pre_activations,
                candidate_states,
                input_gates,
                jacobian_diagonals,
                hidden_states,
            ),
        ) = jax.lax.scan(
            step,
            (initial_hidden_state, initial_cell_state),
            input_sequence.swapaxes(0, 1),
        )
        # Swap back to [B, T, ...] format
        output = outputs.swapaxes(0, 1)
        if not return_features:
            return output
        features = LSTMRuntimeTensors(
            candidate_pre_activations=candidate_pre_activations.swapaxes(0, 1),
            candidate_states=candidate_states.swapaxes(0, 1),
            input_gates=input_gates.swapaxes(0, 1),
            hidden_states=hidden_states.swapaxes(0, 1),
            nonlinearity_jacobian_diag=jacobian_diagonals.swapaxes(0, 1),
        )
        return output, features

    def analyze_batch(
        self,
        params: Any,
        x: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, LSTMRuntimeTensors, JacobianFeatureSummary]:
        """
        Runs forward pass and computes Jacobian-based statistics for analysis.
        For LSTM, the effective Jacobian combines candidate state derivative with input gate.
        """
        outputs, runtime_tensors = self.apply(params, x, mask, return_features=True)
        # Effective Jacobian uses candidate path: diag(input_gate * (1 - candidate_state^2)) @ W_hc
        # The input gate has already scaled the candidate in the cell update; include it explicitly
        hidden_jacobian_diagonal = (
            runtime_tensors.nonlinearity_jacobian_diag * runtime_tensors.input_gates
        )
        hidden_to_hidden_weight = params["whc"]["w"]
        jacobian_statistics = compute_jacobian_features(
            hidden_jacobian_diagonal, hidden_to_hidden_weight, mask
        )
        return outputs, runtime_tensors, jacobian_statistics


class UnitaryRNN(BaseSequenceModel):
    """
    RNN with orthogonal (unitary) hidden-to-hidden weight matrix.
    The orthogonal constraint helps maintain gradient norm and improve training stability.
    """

    def __init__(
        self,
        config: ModelConfig,
        nonlinearity: str = "tanh",
        nonlinearity_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Initializes UnitaryRNN with specified configuration and activation function.
        """
        super().__init__(config)
        self._nonlinearity: Nonlinearity = get_nonlinearity(
            nonlinearity, **(nonlinearity_kwargs or {})
        )
        self.param_dtype = _resolve_dtype(config.param_dtype or config.precision)
        self.use_layer_norm = config.use_layer_norm

    def _orthogonal_matrix(self, raw_weight_matrix: Array) -> Array:
        """
        Converts a raw weight matrix to an orthogonal matrix using QR decomposition.
        Ensures the matrix has determinant +1 by adjusting signs of Q columns.
        """
        base_matrix = raw_weight_matrix.astype(jnp.float32)
        orthogonal_matrix, upper_triangular_matrix = jnp.linalg.qr(base_matrix)
        diagonal_signs = jnp.sign(jnp.diag(upper_triangular_matrix))
        # Ensure no zero signs (replace with 1.0)
        diagonal_signs = jnp.where(diagonal_signs == 0.0, 1.0, diagonal_signs)
        return (orthogonal_matrix * diagonal_signs).astype(self.param_dtype)

    def initialize(self, key: jax.Array) -> Any:
        """
        Initializes UnitaryRNN parameters: a raw hidden-to-hidden weight matrix
        (which will be made orthogonal), plus input and output projection weights.
        """
        hidden_to_hidden_key, input_to_hidden_key, hidden_to_output_key, _ = (
            jax.random.split(key, 4)
        )
        return {
            "wh_raw": jax.random.normal(
                hidden_to_hidden_key,
                (self.config.hidden_dim, self.config.hidden_dim),
                dtype=self.param_dtype,
            ),
            "wx": _init_linear(
                input_to_hidden_key,
                self.config.input_dim,
                self.config.hidden_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
            "wo": _init_linear(
                hidden_to_output_key,
                self.config.hidden_dim,
                self.config.output_dim,
                use_bias=True,
                dtype=self.param_dtype,
            ),
        }

    def apply(
        self,
        params: Any,
        x: jnp.ndarray,
        mask: jnp.ndarray,
        *,
        return_features: bool = False,
    ) -> Tuple[jnp.ndarray, RNNRuntimeTensors] | jnp.ndarray:
        """
        Forward pass through the UnitaryRNN.
        Uses an orthogonal hidden-to-hidden weight matrix to help with gradient stability.
        """
        batch_size, _, _ = x.shape
        hidden_dim = self.config.hidden_dim
        input_sequence = x.astype(self.param_dtype)
        initial_hidden_state = jnp.zeros(
            (batch_size, hidden_dim), dtype=self.param_dtype
        )
        # Convert raw weight matrix to orthogonal matrix
        hidden_weight_matrix = self._orthogonal_matrix(params["wh_raw"])
        activation_function = self._nonlinearity.fn
        jacobian_function = self._nonlinearity.jacobian_diag

        def step(carry, current_input):
            """
            Single UnitaryRNN step: combines input projection with orthogonal hidden transformation.
            """
            previous_hidden_state = carry
            # Combine input projection and orthogonal hidden-to-hidden transformation
            pre_activation = _linear_apply(current_input, params["wx"]) + (
                previous_hidden_state @ hidden_weight_matrix.T
            )
            if self.use_layer_norm:
                pre_activation = _layer_norm(pre_activation)
            current_hidden_state = activation_function(pre_activation)
            output_logits = _linear_apply(current_hidden_state, params["wo"])
            jacobian_diagonal = jacobian_function(pre_activation)
            return current_hidden_state, (
                output_logits,
                pre_activation,
                current_hidden_state,
                jacobian_diagonal,
            )

        # Process sequence: input is [T, B, D], output is [T, B, ...]
        _, (outputs, pre_activations, hidden_states, jacobian_diagonals) = jax.lax.scan(
            step, initial_hidden_state, input_sequence.swapaxes(0, 1)
        )
        # Swap back to [B, T, ...] format
        output = outputs.swapaxes(0, 1)
        if not return_features:
            return output
        features = RNNRuntimeTensors(
            pre_activations=pre_activations.swapaxes(0, 1),
            hidden_states=hidden_states.swapaxes(0, 1),
            nonlinearity_jacobian_diag=jacobian_diagonals.swapaxes(0, 1),
        )
        return output, features

    def analyze_batch(
        self,
        params: Any,
        x: jnp.ndarray,
        mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, RNNRuntimeTensors, JacobianFeatureSummary]:
        """
        Runs forward pass and computes Jacobian-based statistics for analysis.
        Uses the orthogonal hidden-to-hidden weight matrix for statistics computation.
        """
        outputs, runtime_tensors = self.apply(params, x, mask, return_features=True)
        hidden_to_hidden_weight = self._orthogonal_matrix(params["wh_raw"])
        jacobian_statistics = compute_jacobian_features(
            runtime_tensors.nonlinearity_jacobian_diag, hidden_to_hidden_weight, mask
        )
        return outputs, runtime_tensors, jacobian_statistics
