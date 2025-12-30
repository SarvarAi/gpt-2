import torch
from torch import nn

from config import GPTConfig


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # initializing class variables
        self.d_out = d_out
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length
        self.qkv_bias = qkv_bias
        self.head_dim = d_out // num_heads

        # Validation
        assert self.d_out % self.num_heads == 0, "d_out must be divisible by num_heads"

        # components
        self.Q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.K = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.V = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_linear = nn.Linear(d_out, d_out)
        self.out_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch, n_tokens, d_in = x.shape

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        k = k.view(batch, n_tokens, self.num_heads, self.head_dim)
        v = v.view(batch, n_tokens, self.num_heads, self.head_dim)
        q = q.view(batch, n_tokens, self.num_heads, self.head_dim)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = q @ k.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:n_tokens, :n_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(batch, n_tokens, self.d_out)
        context_vec = self.out_linear(context_vec)  # optional projection

        return context_vec


class TransformerBlock(nn.Module):
    def __init__(self, d_in, d_out, context_length, n_heads, dropout, qkv_bias):
        super().__init__()

        self.att = MultiHeadAttention(
            d_in=d_in,
            d_out=d_out,
            context_length=context_length,
            num_heads=n_heads,
            dropout=dropout,
            qkv_bias=qkv_bias,
        )

        self.ff = FeedForward(d_in)
        self.norm1 = LayerNorm(d_in)
        self.norm2 = LayerNorm(d_in)
        self.drop_shortcut = nn.Dropout(dropout)

    def forward(self, x):
        # Shortcut connection for attention block

        shortcut = x

        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                d_in=cfg.emb_dim,
                d_out=cfg.emb_dim,
                context_length=cfg.context_length,
                n_heads=cfg.n_heads,
                dropout=cfg.drop_rate,
                qkv_bias=cfg.qkv_bias,
            ) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
