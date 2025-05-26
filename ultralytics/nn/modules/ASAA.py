# import torch
# import torch.nn as nn
# from timm.layers import trunc_normal_
# from einops import repeat
# import math
# from ultralytics.nn.modules.conv import Conv
# from ultralytics.nn.modules.block import C2f, C3
#
#
# class LinearProjection(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.to_q = nn.Linear(dim, inner_dim, bias=bias)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
#         self.dim = dim
#         self.inner_dim = inner_dim
#
#     def forward(self, x, attn_kv=None):
#         B_, N, C = x.shape
#         if attn_kv is not None:
#             attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
#         else:
#             attn_kv = x
#         N_kv = attn_kv.size(1)
#         q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
#         kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
#         q = q[0]
#         k, v = kv[0], kv[1]
#         return q, k, v
#
#
# ########### window-based self-attention #############
# class WindowAttention_ASSA(nn.Module):
#     def __init__(self, dim, win_size=(8, 8), num_heads=4, token_projection='linear', qkv_bias=True, qk_scale=None,
#                  attn_drop=0.,
#                  proj_drop=0.):
#
#         super().__init__()
#         self.dim = dim
#         self.win_size = win_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
#         coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.win_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)
#         trunc_normal_(self.relative_position_bias_table, std=.02)
#
#         if token_projection == 'linear':
#             self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
#         else:
#             raise Exception("Projection error!")
#
#         self.token_projection = token_projection
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.relu = nn.ReLU()
#         self.w = nn.Parameter(torch.ones(2))
#
#     def forward(self, x, attn_kv=None, mask=None):
#         # 调整输入维度，从 (B, C, H, W) 转为 (B, H, W, C)
#
#         x = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])
#         B_, N, C = x.shape
#         q, k, v = self.qkv(x, attn_kv)
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         ratio = attn.size(-1) // relative_position_bias.size(-1)
#         relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)
#
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             nW = mask.shape[0]
#             mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N * ratio)
#             attn0 = self.softmax(attn)
#             attn1 = self.relu(attn) ** 2  # b,h,w,c
#         else:
#             attn0 = self.softmax(attn)
#             attn1 = self.relu(attn) ** 2
#         w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
#         w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
#         attn = attn0 * w1 + attn1 * w2
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         x = x.reshape(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), x.shape[2]).permute(0, 3, 1,
#                                                                                                               2)
#         return x
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'
#
#
# class Bottleneck_WAtt_ASSA(nn.Module):
#     """Standard bottleneck."""
#
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.cv3 = WindowAttention_ASSA(dim=c2)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         """Applies the YOLO FPN to input data."""
#         return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
#
#
# class C3k(C3):
#     """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
#
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
#         """Initializes the C3k module with specified channels, number of layers, and configurations."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#         self.m = nn.Sequential(*(Bottleneck_WAtt_ASSA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
#
#
# class C3k2_WAtt_ASSA(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""
#
#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WAtt_ASSA(self.c, self.c, shortcut, g) for _ in
#             range(n)
#         )
#
#
# if __name__ == '__main__':
#     ASSA_Attention = WindowAttention_ASSA(dim=256)
#     # 创建一个输入张量
#     batch_size = 1
#     input_tensor = torch.randn(batch_size, 256, 64, 64)
#     # 运行模型并打印输入和输出的形状
#     output_tensor = ASSA_Attention(input_tensor)
#     print("Input shape:", input_tensor.shape)
#     print("0utput shape:", output_tensor.shape)


import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import repeat
import math
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, C3


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v


class WindowAttention_ASSA(nn.Module):
    def __init__(self, dim, win_size=(8, 8), num_heads=4, token_projection='linear', qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # Get pairwise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        # Adjust input dimensions from (B, C, H, W) to (B, H*W, C)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2] * x.shape[3], x.shape[1])  # (B, N, C)
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Ensure the relative position bias is aligned with attention map
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww, Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)  # Scale factor for upsampling the bias
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        # Debugging print statements for checking shapes
        print(f"attn shape before adding bias: {attn.shape}")
        print(f"relative_position_bias shape before reshape: {relative_position_bias.shape}")

        # Resize relative_position_bias to match attn size
        # We need to reshape relative_position_bias to [num_heads, N, N]
        relative_position_bias = relative_position_bias.view(self.num_heads, N, N)

        print(f"relative_position_bias shape after reshape: {relative_position_bias.shape}")

        # Add the relative position bias to attention map
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2

        # Combine the attention maps with weighted sum
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        # Apply attention to values and project back to original dimension
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back to (B, C, H, W)
        x = x.reshape(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), x.shape[2]).permute(0, 3, 1, 2)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'


class Bottleneck_WAtt_ASSA(nn.Module):
    """Standard bottleneck with Window Attention and ASSA"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = WindowAttention_ASSA(dim=c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_WAtt_ASSA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_WAtt_ASSA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck_WAtt_ASSA(self.c, self.c, shortcut, g) for _ in
            range(n)
        )


if __name__ == '__main__':
    # Test the WindowAttention_ASSA module
    ASSA_Attention = WindowAttention_ASSA(dim=256)
    batch_size = 1
    input_tensor = torch.randn(batch_size, 256, 64, 64)
    output_tensor = ASSA_Attention(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
