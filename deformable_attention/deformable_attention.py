import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

# main class

class DeformableAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        r = 4,
        offset_groups = None,
        offset_kernel_size = 5,
    ):
        super().__init__()
        offset_groups = default(offset_groups, heads)
        assert divisible_by(offset_groups, heads) or divisible_by(heads, offset_groups)

        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        offset_dims = inner_dim // offset_groups

        self.to_offset = nn.Sequential(
            nn.Conv2d(offset_dims, offset_dims, 1, groups = offset_dims, stride = r),
            nn.GELU(),
            nn.Conv2d(offset_dims, 2, 1, bias = False)
        )

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        return x
