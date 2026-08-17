"""
---
title: Utility functions for DDPM experiment
summary: >
  Utility functions for DDPM experiment
---

# Utility functions for [DDPM](index.html) experiemnt
"""
import torch.utils.data


# def gather(consts: torch.Tensor, t: torch.Tensor):
#     """Gather consts for $t$ and reshape to feature map shape"""
#     c = consts.gather(-1, t)
#     return c.reshape(-1, 1, 1)

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    # print(f"shape of t in gather: {t.shape}, dtype: {t.dtype}")
    
    # ✅ Ensure t is an integer tensor
    if not torch.is_floating_point(t) and not t.dtype == torch.long:
        t = t.long()

    # ✅ Ensure all values in t are within valid range
    if torch.any(t < 0) or torch.any(t >= consts.shape[-1]):
        raise ValueError(f"❌ Invalid index detected in t: {t} (Max index: {consts.shape[-1] - 1})")
    
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)
