from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool


# setting the GPT configuration
GPT_CONFIG = GPTConfig(
    vocab_size=50_257,
    context_length=1_024,
    emb_dim=12 * 64,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False
)
