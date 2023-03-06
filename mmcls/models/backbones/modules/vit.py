import torch
from torch import nn
from mmcls.models.vit.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, head_num, token_num, token_num]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HeadFusionAttention(nn.Module):
    """
    fuse front head output add current head input as current input
    Origin: head2_output = f(head2_input)
    Fuse: head2_output = f(head1_output + head2_input)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.group_number = 4 # 每个group内部并行计算
        self.qkv = nn.ModuleList([nn.Linear(dim//self.group_number, (dim//self.group_number)*3, bias=qkv_bias) for _ in range(self.group_number)])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # H = self.num_heads
        # dim = C // self.num_heads
        g = self.group_number
        g_dim = C // self.group_number
        x = x.reshape((B, N, g, g_dim))

        outputs = []
        head_x = torch.zeros((B, N, g_dim), device=x.device)
        for i in range(g):
            # self-attention
            current_x = x[:, :, i]
            current_x = current_x + head_x
            qkv = self.qkv[i](current_x).reshape(B, N, 3, g_dim)
            qkv = qkv.permute(2, 0, 1, 3)   # [3, B, N, dim]
            q, k, v = qkv[0], qkv[1], qkv[2] 
            attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            head_x = (attn @ v)  # [B, N, d]
            outputs.append(head_x)
        x = torch.cat(outputs, dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HeadFusionAttentionV2(nn.Module):
    """
    V2:  one branch is origin, onther is modified
    fuse front head output add current head input as current input
    Origin: head2_output = f(head2_input)
    Fuse: head2_output = f(head1_output + head2_input)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.group_number = 2 # 每个group内部并行计算.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv2 = nn.ModuleList([nn.Linear(dim//self.group_number, (dim//self.group_number)*3, bias=qkv_bias) for _ in range(self.group_number)])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # original        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, head_num, token_num, token_num]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        origin = (attn @ v)     # [B, H, N, dim]

        # modify
        g = self.group_number
        g_dim = C // self.group_number
        x = x.reshape((B, N, g, g_dim))
        n = self.num_heads // g  # head number per group

        origin = origin.transpose(1, 2).reshape((B, N, g, g_dim))

        outputs = []
        head_x = torch.zeros((B, N, g_dim), device=x.device)
        for i in range(g):
            # self-attention
            current_x = x[:, :, i]
            current_x = current_x + head_x
            qkv = self.qkv2[i](current_x).reshape(B, N, 3, g_dim)
            qkv = qkv.permute(2, 0, 1, 3)   # [3, B, N, dim]
            q, k, v = qkv[0], qkv[1], qkv[2] 
            attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, N, N]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            head_x = (attn @ v)  # [B, N, d]
            outputs.append(head_x + origin[:, :, i])
        x = torch.cat(outputs, dim=-1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x





class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_fusion=False,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if head_fusion:
            self.attn = HeadFusionAttentionV2(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        feature = self.attn(self.norm1(x))
        x = x + self.drop_path(feature)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

