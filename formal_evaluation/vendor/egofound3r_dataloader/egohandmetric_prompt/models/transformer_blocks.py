from __future__ import annotations

from torch import nn


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, key_padding_mask=None):
        residual = self.norm1(x)
        attn_out, _ = self.attn(
            residual,
            residual,
            residual,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, memory, self_key_padding_mask=None, memory_key_padding_mask=None):
        residual = self.self_norm(x)
        self_out, _ = self.self_attn(
            residual,
            residual,
            residual,
            key_padding_mask=self_key_padding_mask,
            need_weights=False,
        )
        x = x + self_out

        cross_query = self.cross_norm(x)
        cross_out, _ = self.cross_attn(
            cross_query,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + cross_out
        return x + self.mlp(self.mlp_norm(x))
