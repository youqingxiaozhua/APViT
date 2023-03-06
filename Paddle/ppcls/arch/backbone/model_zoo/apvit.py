# from x2paddle import torch2paddle
import math
import random
import logging
from itertools import repeat
import paddle
from functools import partial
import warnings
from typing import Tuple
import numpy as np
from paddle import nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal
from .irse_v2 import IRSEV2
from .la_max import LANet


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


trunc_normal_ = TruncatedNormal(std=.02)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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


class Attention(nn.Layer):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C // self.num_heads))
        if hasattr(self, 'head_drop'):
            qkv = self.head_drop(qkv)
        qkv = qkv.transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        # attn = q @ k.transpose(-2, -1) * self.scale
        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        # attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn.matmul(v)).transpose((0, 2, 1, 3))
        x = x.reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn
        .GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def top_pool(x, dim=1, keep_num:int=None, keep_rate=None, exclude_first=True, **kwargs):
    """
    根据输入x 的值和方差来选择 topk 个元素，并返回其 index, index的shape为[B, keep_num, dim]
    选择标准为 alpha1 * mean + alpha2 * std
    args:
        exclude_first: if set to True, will return the index of the first element and the top k-1 elements
    """
    assert x.ndim == 3, 'input x must have 3 dimensions(B, N, C)'
    assert not (keep_num is not None and keep_rate is not None), 'keep_num and keep_rate can not be assigned on the same time'
    assert not (keep_num is None and keep_rate is None)
    B, N, C = x.shape
    if exclude_first is True:
        x = x[:, 1:, :]
        N -= 1
    if keep_num is None:
        keep_num = max(int(N * keep_rate), 1)
    
    if N == keep_num:
        return None

    pool_weight = x.sum(axis=-1)

    if exclude_first is False:
        try:
            _, keep_index = paddle.topk(pool_weight, k=keep_num, axis=1, sorted=False)
        except Exception as e:
            print(e)
            print('pool_weight', pool_weight.shape)
            print('k', keep_num)
            exit()
        keep_index = paddle.sort(keep_index, axis=1)
    else:
        # pool_weight = pool_weight[:, 1:, ...]
        _, keep_index = paddle.topk(pool_weight, k=keep_num, axis=1, sorted=False)
        keep_index = paddle.sort(keep_index, axis=1)
        keep_index = paddle.concat([
            # paddle.zeros([B, 1]).type(paddle.int16).to(keep_index.device),
            paddle.zeros([B, 1], dtype=keep_index.dtype),
            keep_index + 1], axis=1)
    return keep_index


class PoolingAttention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 pool_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool_config = pool_config


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape((B, N, 3, self.num_heads, C // self.num_heads))
        qkv = qkv.transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale  # [B, head_num, token_num, token_num]

        if self.pool_config:
            attn_method = self.pool_config.get('attn_method')
            if attn_method == 'SUM_ABS_1':
                attn_weight = attn[:, :, 0, :].transpose((0, 2, 1))    # [B, token_num, head_num]
                attn_weight = paddle.sum(paddle.abs(attn_weight), axis=-1).unsqueeze(-1)
            elif attn_method == 'SUM':
                attn_weight = attn[:, :, 0, :].transpose((0, 2, 1))    # [B, token_num, head_num]
                attn_weight = paddle.sum(attn_weight, axis=-1).unsqueeze(-1)
            elif attn_method == 'MAX':
                attn_weight = attn[:, :, 0, :].transpose((0, 2, 1))
                attn_weight = paddle.max(attn_weight, axis=-1)[0].unsqueeze(-1)
            else:
                raise ValueError('Invalid attn_method: %s' % attn_method)
            keep_index = top_pool(attn_weight, dim=self.dim, **self.pool_config)
        else:
            keep_index = None

        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3))
        x = x.reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, keep_index


class PoolingBlock(nn.Layer):

    def __init__(self, dim=0, num_heads=0, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_config=None, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            pool_config=pool_config)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        feature, keep_index = self.attn(self.norm1(x))
        x = x + self.drop_path(feature)
        if keep_index is not None:
            if len(keep_index) != x.shape[1]:
                # x = x.gather(dim=1, index=keep_index)
                pooled_x = []
                for i in range(keep_index.shape[0]):
                    pooled_x.append(x[i][keep_index[i]])
                x = paddle.stack(pooled_x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=\
        768, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[0] + 2 * padding - patch_size[0]) / stride + 1
        if num_patches % 1 != 0:
            print('Warning: patch numbser is not int:', num_patches)
        num_patches = int(num_patches) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size,
            stride=stride, padding=padding)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1
            ], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Layer):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=\
        3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, paddle.nn.Layer)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with paddle.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(paddle.zeros([1, in_chans, img_size[0],
                    img_size[1]]).requires_grad_(False))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:
            ]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    print('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2
        )
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode=\
        'bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new *
        gs_new, -1)
    # posemb = torch2paddle.concat([posemb_tok, posemb_grid], dim=1)
    posemb = paddle.concat([posemb_tok, posemb_grid], axis=1)
    return posemb


class PoolingViT(nn.Layer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=\
        768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, 
        drop_path_rate=0.0, hybrid_backbone=None, norm_layer_eps=1e-05,
        patch_stride=None, patch_padding=0, freeze=False, input_type=\
        'image', num_patches=None, pretrained=None, 
        in_channels=[256],
        attn_method='SUM_ABS_1',
        cnn_pool_config=None,
        vit_pool_configs=None,
        **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type in ('image', 'feature'
            ), 'input_type must be "image" or "feature"'
        if input_type == 'feature':
            assert num_patches is not None
        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, epsilon=norm_layer_eps)
        self.num_features = self.embed_dim = embed_dim
        if input_type == 'image':
            if hybrid_backbone is not None:
                self.patch_embed = HybridEmbed(hybrid_backbone, img_size=\
                    img_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(img_size=img_size, patch_size
                    =patch_size, in_chans=in_chans, embed_dim=embed_dim,
                    stride=patch_stride, padding=patch_padding)
            num_patches = self.patch_embed.num_patches

        self.cls_token = self.create_parameter(shape=(1, 1, embed_dim))
        self.add_parameter("cls_token", self.cls_token)
        self.patch_pos_embed = self.create_parameter(shape=(1, num_patches, embed_dim))
        self.add_parameter("patch_pos_embed", self.patch_pos_embed)
        self.cls_pos_embed = self.create_parameter(shape=(1, 1, embed_dim))
        self.add_parameter("cls_pos_embed", self.cls_pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)


        self.projs = nn.LayerList([nn.Conv2D(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config

        if attn_method == 'LA':
            self.attn_f = LANet(embed_dim, 16)
        elif attn_method == 'SUM':
            self.attn_f = lambda x: paddle.sum(x, axis=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: paddle.sum(paddle.abs(x), axis=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: paddle.sum(paddle.pow(paddle.abs(x), 2), axis=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: paddle.max(x, axis=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: paddle.max(paddle.abs(x), axis=1)[0].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = np.linspace(0, drop_path_rate, depth)
        if vit_pool_configs is None:
            self.blocks = nn.LayerList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    )
                for i in range(depth)])
        else:
            vit_keep_rates = vit_pool_configs['keep_rates']
            self.blocks = nn.LayerList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_keep_rates[i], **vit_pool_configs),
                    )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.patch_pos_embed)
        trunc_normal_(self.cls_pos_embed)
        trunc_normal_(self.cls_token)
        # self.apply(self._init_weights)


    def forward_features(self, x):
        assert len(x) == 1, len(x)
        x = [self.projs[i](x[i]) for i in range(len(x))]

        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient
        x = [i.flatten(2).transpose((0, 2, 1)) for i in x]
        # x = self.projs[0](x).flatten(2).transpose(2, 1)

        attn_weight = attn_map.flatten(2).transpose((0, 2, 1))
        
        x = paddle.stack(x).sum(axis=0)   # S1 + S2 + S3

        x = x + self.patch_pos_embed
        
        B, N, C = x.shape
        
        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            if keep_indexes is not None:
                pooled_x = []
                for i in range(keep_indexes.shape[0]):
                    pooled_x.append(x[i][keep_indexes[i]])
                x = paddle.stack(pooled_x)
                # x = x.gather_nd(index=keep_indexes)
        cls_tokens = self.cls_token.expand((B, -1, -1))  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        x = paddle.concat((cls_tokens, x), axis=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        return x


class LinearClsHead(nn.Layer):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss.
    """

    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        # self.loss = nn.CrossEntropyLoss()
        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        self._init_layers()

    def _init_layers(self):
        self.fc = nn.Linear(self.in_channels, self.num_classes, weight_attr=nn.initializer.Constant(value=0.))

    # def init_weights(self):
    #     constant_init(self.fc, val=0, bias=0)

    def forward(self, x, gt_label=None):
        cls_score = self.fc(x)
        # if gt_label is not None:
        #     losses = self.loss(cls_score, gt_label)
        #     return losses
        # else:
        #     return cls_score
        return cls_score


class APViT(nn.Layer):
    """Attentive Pooling ViT"""

    def __init__(self, class_num=7):
        super().__init__()
        self.extractor = IRSEV2(input_size=(112, 112), num_layers=44, mode='ir', return_index=(2,))
        self.vit = PoolingViT(input_type='feature', num_patches=196,
            embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-06,
            in_channels=[256],
            attn_method='SUM_ABS_1',    # CNN Attention method， SUM_ABS_1
            cnn_pool_config=dict(keep_num=160, exclude_first=False),
            vit_pool_configs=dict(keep_rates=[1.] * 6 + [0.9] * 6, exclude_first=True, attn_method='SUM')
            )
        self.head = LinearClsHead(num_classes=class_num, in_channels=768)
        # load_pretrain
        data = paddle.load('weights/ir.pdparams')
        self.extractor.set_state_dict(data)
        data = paddle.load('weights/vit.pdparams')
        self.vit.set_state_dict(data)

    def forward(self, img, gt_label=None):
        x = self.extractor(img)
        x = self.vit(x)
        x = self.head(x, gt_label)
        return x


if __name__ == '__main__':
    model = APViT()
    # ckpt = paddle.load(
    #     '../RAFa_2_6131_ep33_90p91_param.pdiparams')
    # for k in list(ckpt.keys()):
    #     if k.split('.')[-1] == 'weight':
    #         new_key = k[:-6] + '_weight'
    #         ckpt[new_key] = ckpt[k]
    # # print(ckpt.keys())
    # model.load_state_dict(ckpt)
    model.eval()
    mock_input = paddle.rand((3, 3, 112, 112))
    with paddle.no_grad():
        out = model(mock_input)
    print(out.shape)
