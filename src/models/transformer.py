from __future__ import annotations

from typing import Any, Dict, List

import jax
import jax.numpy as jnp

from .base import BaseSequenceModel, ModelConfig

# Attempt to locate a Modula Transformer/Attention implementation
_MODULA_AVAILABLE = False
_ModulaTransformer = None

try:
    # Potential paths (update if repository exposes a different API)
    from modula.transformers import Transformer as _MT  # type: ignore

    _ModulaTransformer = _MT
    _MODULA_AVAILABLE = True
except Exception:
    try:
        from modula.attention import Transformer as _MT  # type: ignore

        _ModulaTransformer = _MT
        _MODULA_AVAILABLE = True
    except Exception:
        try:
            from modula.attention import Attention  # type: ignore

            # Might build stack around Attention if needed
            _MODULA_AVAILABLE = True
        except Exception:
            _MODULA_AVAILABLE = False

try:
    from modula.atom import Linear
    from modula.bond import ReLU

    _LINEAR_AVAILABLE = True
except Exception:
    _LINEAR_AVAILABLE = False


class TransformerAdapter(BaseSequenceModel):
    """Adapter over Modula transformer with a robust native JAX fallback.

    Order of preference:
    1) Use Modula's Transformer if importable
    2) Use native JAX causal multi-head self-attention stack
    3) Fallback to simple MLP if Modula Linear available
    """

    def __init__(
        self, config: ModelConfig, num_heads: int = 4, num_layers: int | None = None
    ):
        super().__init__(config)
        self.num_heads = num_heads
        self.num_layers = num_layers or config.num_layers
        self._arch = None
        self._mode: str = "auto"  # "modula" | "native" | "mlp"

    def _build_architecture(self):
        # Prefer direct Modula Transformer
        if _ModulaTransformer is not None:
            self._arch = _ModulaTransformer(
                input_dim=self.config.input_dim,
                model_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
            )
            if hasattr(self._arch, "jit"):
                self._arch.jit()
            self._mode = "modula"
            return
        # Next: use native JAX transformer
        self._mode = "native"
        # If native path is used, params will be created in initialize()

    def _native_init(self, key: jax.Array) -> Dict[str, Any]:
        D_in = self.config.input_dim
        D_model = self.config.hidden_dim
        D_out = self.config.output_dim
        H = max(1, self.num_heads)
        head_dim = max(1, D_model // H)
        # If not divisible, last head will handle the remainder via projection
        keys = jax.random.split(key, 3 + 6 * self.num_layers)
        k_in, k_out = keys[0], keys[1]
        ptr = 2

        def lin(shape, k, scale=1.0):
            return jax.random.normal(k, shape) * (scale / jnp.sqrt(shape[0]))

        params: Dict[str, Any] = {}
        params["proj_in_W"] = lin((D_in, D_model), k_in)
        params["proj_in_b"] = jnp.zeros((D_model,))
        layers: List[Dict[str, jnp.ndarray]] = []
        for _ in range(self.num_layers):
            kq, kk, kv, ko, kf1, kf2 = keys[ptr : ptr + 6]
            ptr += 6
            layer = {
                "Wq": lin((D_model, H * head_dim), kq),
                "Wk": lin((D_model, H * head_dim), kk),
                "Wv": lin((D_model, H * head_dim), kv),
                "Wo": lin((H * head_dim, D_model), ko),
                "Wff1": lin((D_model, 4 * D_model), kf1),
                "bff1": jnp.zeros((4 * D_model,)),
                "Wff2": lin((4 * D_model, D_model), kf2),
                "bff2": jnp.zeros((D_model,)),
            }
            layers.append(layer)
        params["layers"] = layers
        params["proj_out_W"] = lin((D_model, D_out), k_out)
        params["proj_out_b"] = jnp.zeros((D_out,))
        return params

    @staticmethod
    def _causal_mask(T: int) -> jnp.ndarray:
        # [1, 1, T, T] boolean mask with True for valid positions i>=j
        mask = jnp.tril(jnp.ones((T, T), dtype=bool))
        return mask[None, None, :, :]

    @staticmethod
    def _mha(
        x: jnp.ndarray, layer: Dict[str, jnp.ndarray], num_heads: int
    ) -> jnp.ndarray:
        # x: [B, T, D_model]
        B, T, Dm = x.shape
        H = max(1, num_heads)
        head_dim_total = layer["Wq"].shape[1]
        head_dim = head_dim_total // H
        # Projections
        Q = x @ layer["Wq"]
        K = x @ layer["Wk"]
        V = x @ layer["Wv"]

        # Reshape to [B, H, T, head_dim]
        def split_heads(a):
            return a.reshape((B, T, H, head_dim)).transpose(0, 2, 1, 3)

        Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
        # Scaled dot-product attention with causal mask
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_logits = (Qh @ Kh.transpose(0, 1, 3, 2)) * scale  # [B, H, T, T]
        mask = TransformerAdapter._causal_mask(T)
        attn_logits = jnp.where(mask, attn_logits, jnp.full_like(attn_logits, -1e30))
        attn = jax.nn.softmax(attn_logits, axis=-1)
        Oh = attn @ Vh  # [B, H, T, head_dim]
        # Merge heads
        O = Oh.transpose(0, 2, 1, 3).reshape((B, T, H * head_dim))
        # Output projection
        return O @ layer["Wo"]

    @staticmethod
    def _ffn(x: jnp.ndarray, layer: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        h = x @ layer["Wff1"] + layer["bff1"]
        h = jax.nn.gelu(h)
        return h @ layer["Wff2"] + layer["bff2"]

    def _native_apply(self, params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
        # x: [B, T, D_in]
        x = x @ params["proj_in_W"] + params["proj_in_b"]
        for layer in params["layers"]:
            attn_out = TransformerAdapter._mha(x, layer, self.num_heads)
            x = x + attn_out
            ffn_out = TransformerAdapter._ffn(x, layer)
            x = x + ffn_out
        return x @ params["proj_out_W"] + params["proj_out_b"]

    def _build_mlp_fallback(self):
        if not _LINEAR_AVAILABLE:
            raise RuntimeError(
                "Modula Linear not available; install from https://github.com/modula-systems/modula.git"
            )
        mlp = Linear(self.config.input_dim, self.config.hidden_dim)
        mlp @= ReLU()
        mlp @= Linear(self.config.hidden_dim, self.config.hidden_dim)
        mlp @= ReLU()
        mlp @= Linear(self.config.hidden_dim, self.config.output_dim)
        mlp.jit()
        self._arch = mlp
        self._mode = "mlp"

    def initialize(self, key: jax.Array) -> Any:
        if self._arch is None and self._mode == "auto":
            self._build_architecture()
        if self._mode == "modula":
            return self._arch.initialize(key)
        elif self._mode == "native":
            return self._native_init(key)
        else:
            # Try to build MLP fallback if not already built
            if self._arch is None:
                self._build_mlp_fallback()
            return self._arch.initialize(key)

    def apply(self, params: Any, x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        # Predict per-time-step output; we keep shape [B, T, D_out]
        B, T, D = x.shape
        if self._mode == "modula":
            xt = x.reshape((B * T, D))
            pred = self._arch.apply(params, xt)
            return pred.reshape((B, T, -1))
        elif self._mode == "native":
            return self._native_apply(params, x)
        else:
            xt = x.reshape((B * T, D))
            pred = self._arch.apply(params, xt)
            return pred.reshape((B, T, -1))
