import math
from typing import Optional, Tuple, Union, List
import torch.nn.functional as F
import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = Swish()
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb



class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.1):
        super().__init__()
        
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)  
        
        self.act1 = Swish()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_channels)  
        self.act2 = Swish()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # If in_channels and out_channels do not match, apply a 1x1 conv for the shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Time embedding layer and activation
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # print(f"ResidualBlock Input shape: {x.shape} and the shape of t is {t.shape}")
        # print(f"self.norm1(x) shape {self.norm1(x).shape}")
        # First convolution block with time embedding
        h = self.conv1(self.act1(self.norm1(x)))
        
        # print(f"Time tensor shape before adding: {t.shape}")
        h += self.time_emb(self.time_act(t))[:, :, None]  # Broadcasting time embedding
#         print(f"Shape after adding time embedding: {h.shape}")
#         
#         print(f"After Conv1: {h.shape}")
        
        # Second convolution block with dropout
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        # print(f"After Conv2: {h.shape}")
        
        # Adding the shortcut connection (skip connection)
        return h + self.shortcut(x)



class AttentionBlock(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        super().__init__()

        if d_k is None:
            d_k = n_channels
        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5

        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        # print(f"AttentionBlock Input shape: {x.shape}")
        batch_size, n_channels, length = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        attn = attn.softmax(dim=2)
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        res = self.output(res)
        res += x

        res = res.permute(0, 2, 1).view(batch_size, n_channels, length)
        # print(f"AttentionBlock Output shape: {res.shape}")
        return res


class DownBlock(nn.Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # print(f"DownBlock Input shape: {x.shape}")
        x = self.res(x, t)
        x = self.attn(x)
        # print(f"DownBlock Output shape: {x.shape}")
        return x



class UpBlock(nn.Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        
        # Combine the output channels after upsampling
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # print(f"UpBlock Input shape: {x.shape}")
        
        # Residual block with concatenated x and output channels
        x = self.res(x, t)
        # print(f"After res in upBlock: {x.shape}")
        
        # Apply attention if required
        x = self.attn(x)
        
        # print(f"UpBlock Output shape: {x.shape}")
        
        return x




class MiddleBlock(nn.Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose1d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`
        # print(f"Upsample Input shape: {x.shape}")
        _ = t
        x = self.conv(x)
        # print(f"Upsample Output shape: {x.shape}")
        return x





class Downsample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # print(f"Downsample Input shape: {x.shape}")
        _ = t
        x = self.conv(x)
        # print(f"Downsample Output shape: {x.shape}")
        return x




class UNet(nn.Module):
    def __init__(self, image_channels: int = 4, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        n_resolutions = len(ch_mults)

        self.image_proj = nn.Conv1d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(n_channels * 4)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_resolutions):
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        self.down = nn.ModuleList(down)

        self.middle = MiddleBlock(out_channels,  n_channels * 4)

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv1d(in_channels, image_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, None] = None, t_emb:Optional[torch.Tensor] = None):
        """
        x: Input [B, C, T]
        t: Timestep tensor (used if t_emb not provided)
        t_emb: Precomputed time embedding (used in TSG / CFG variants)
        """
        # print(f"Input shape in Unet: {x.shape}")
        # ✅ If t_emb is not provided, compute from t
        if t_emb is None:
            t_emb = self.time_emb(t)  # [B, D]

        x = self.image_proj(x)
        # print(f"After Image Projection: {x.shape}")
        h = [x]

        for m in self.down:
            x = m(x, t_emb)
            h.append(x)

        x = self.middle(x, t_emb)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t_emb)
            else:
                s = h.pop()
                diff = abs(x.shape[2] - s.shape[2])
                if diff == 1:
                    if x.shape[2] < s.shape[2]:
                        x = F.pad(x, (0, 1))
                    else:
                        s = F.pad(s, (0, 1))
                x = torch.cat((x, s), dim=1)
                x = m(x, t_emb)

        x = self.final(self.act(self.norm(x)))
        return x

#     def forward(self, x: torch.Tensor, t: torch.Tensor):
#         
#         # print(f"Input shape in Unet: {x.shape}")
#         
#         # Get time-step embeddings
#         t = self.time_emb(t)
#         
#         # Get image projection
#         x = self.image_proj(x)
#         # print(f"After Image Projection: {x.shape}")
#         
#         # `h` will store outputs at each resolution for skip connection
#         h = [x]
#         
#         # First half of U-Net
#         for m in self.down:
#             x = m(x, t)
#             # print(f"After Down Block: {x.shape}")
#             h.append(x)
# 
#         # Middle (bottom)
#         x = self.middle(x, t)
#         # print(f"After Middle Block: {x.shape}")
# 
#         # print("$\bf{starting second half}$")
#         # Second half of U-Net
#         it=0
#         for m in self.up:
#             # print(f" toatl layers in up {len(self.up)} and it the itteration number {it} \n ")
#             it=it+1
#             if isinstance(m, Upsample):
#                 x = m(x, t)
#             else:
#                 # Get the skip connection from first half of U-Net and concatenate
#                 s = h.pop()
#                 # print(f"x, s,and t shapes {x.shape}, {s.shape}, {t.shape}")
#                 
#                 # diff of x and y shapes:
#                 diff = abs(x.shape[2] - s.shape[2])
#                 
#                 if diff == 1:  # Only apply padding if difference is 1
#                     # print(f"\n \t \t x, s  shapes {x.shape}, {s.shape}")
#                     if x.shape[2] < s.shape[2]:
#                         x = F.pad(x, (0, 1))  # Pad on the last dimension of x
#                     else:
#                         s = F.pad(s, (0, 1))  # Pad on the last dimension of s
#                 
#                 x = torch.cat((x, s), dim=1)
#                 # print(f"x shape after concatinate{x.shape}")
#                 
#                 x = m(x, t)
#             # print(f"After Up Block: {x.shape}")
#             
#         # Final normalization and convolution
#         x = self.final(self.act(self.norm(x)))
#         # print(f"Final Output shape: {x.shape}")
#         return x
    
