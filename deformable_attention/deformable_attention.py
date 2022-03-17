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
        offset_groups = None,
    ):
        super().__init__()
        offset_groups = default(offset_groups, heads)
        assert divisible_by(offset_groups, heads) or divisible_by(heads, offset_groups)

    def forward(self, x):
        return x
