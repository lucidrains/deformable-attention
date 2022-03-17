<img src="./deformable-attention.png" width="500px"></img>

## Deformable Attention

Implementation of Deformable Attention in Pytorch from <a href="https://arxiv.org/abs/2201.00520">this paper</a>, which appears to be better than what was proposed in DETR. This repository may also explore 1d and 3d cases, as well as fix the relative positional bias to extrapolate better (SwinV2 style).

## Install

```bash
$ pip install deformable-attention
```

## Usage

```python
import torch
from deformable_attention import DeformableAttention

attn = DeformableAttention(
    dim = 512,                   # feature dimensions
    dim_head = 64,               # dimension per head
    heads = 8,                   # attention heads
    dropout = 0.,                # dropout
    downsample_factor = 4,       # downsample factor (r in paper)
    offset_scale = 4,            # scale of offset, maximum offset
    offset_groups = None,        # number of offset groups, should be multiple of heads
    offset_kernel_size = 5,      # offset kernel size, 5 was in example in paper
)

x = torch.randn(1, 512, 64, 64)
attn(x) # (1, 512, 64, 64)
```

## Todo

- [ ] complete cross attention variant

## Citation

```bibtex
@misc{xia2022vision,
    title   = {Vision Transformer with Deformable Attention}, 
    author  = {Zhuofan Xia and Xuran Pan and Shiji Song and Li Erran Li and Gao Huang},
    year    = {2022},
    eprint  = {2201.00520},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@misc{liu2021swin,
    title   = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    author  = {Ze Liu and Han Hu and Yutong Lin and Zhuliang Yao and Zhenda Xie and Yixuan Wei and Jia Ning and Yue Cao and Zheng Zhang and Li Dong and Furu Wei and Baining Guo},
    year    = {2021},
    eprint  = {2111.09883},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
