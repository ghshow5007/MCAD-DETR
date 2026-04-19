import torch
import torch.nn as nn
import einops
from .conv import Conv

class EfficientAdditiveAttnetion(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=1):
        super().__init__()
        token_dim = in_dims
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x_4d):
        B, C, H, W = x_4d.size()
        
        x = x_4d.flatten(2).transpose(2, 1)
        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor

        A = torch.nn.functional.normalize(A, dim=1)

        G = torch.sum(A * query, dim=1)

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )

        out = self.Proj(G * key) + query

        out = self.final(out)

        return out.transpose(2, 1).reshape((B, C, H, W))
