# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import logging
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_state_dict

from ..builder import BACKBONES
from ..vit.layers import DropPath, trunc_normal_
from .modules.vit_pooling import PoolingBlock
from .modules.t2t import Token_performer
from .base_backbone import BaseBackbone

# from .token_transformer import Token_transformer
# from .token_performer import Token_performer
# from .transformer_block import Block, get_sinusoid_encoding

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


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


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(  # TODO: is in_dim=dim right?
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TokenAttention(nn.Module):
    def __init__(self, dim, num_heads=8, in_dim=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        # skip connection
        x = v.squeeze(1) + x   # because the original x has different size with current x, use v to do skip connection

        return x


class Token_transformer(nn.Module):

    def __init__(self, dim, in_dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenAttention(
            dim, in_dim=in_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_dim)
        self.mlp = Mlp(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=in_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            #self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            #self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


@BACKBONES.register_module()
class T2T_ViT(BaseBackbone):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, pretrained=None, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
                img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        if pretrained:
            self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def init_weights(self, pretrained):
        logger = logging.getLogger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'state_dict_ema' in state_dict:
            state_dict = state_dict['state_dict_ema']

        load_state_dict(self, state_dict, strict=False, logger=logger)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        assert x.shape[2] == self.img_size  #[B, C, H, W]
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@BACKBONES.register_module()
class T2T_ViTPooling(T2T_ViT):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64, pretrained=None, pool_config=None):

        super().__init__(**{k:v for k, v in locals().items() if k != 'self'})
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if pool_config is not None:
            assert len(pool_config['keep_rates']) == depth, 'num of keep_rates must be equal to depth'
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=pool_config['keep_rates'][i], **pool_config),
                    )
                for i in range(depth)])


def T2t_vit_7(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']

    return model


def T2t_vit_10(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']

    return model


def T2t_vit_12(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']

    return model


def T2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']

    return model

def T2t_vit_19(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']

    return model


def T2t_vit_24(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']

    return model


def T2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']

    return model


def T2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']

    return model


def T2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']

    return model


# rexnext and wide structure
def T2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']

    return model


def T2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']

    return model
