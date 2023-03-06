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
from .irse import IRSE
from .la_max import MultiLANet, MaxHybridFlatten


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


class HeadDropOut(nn.Layer):
    """
    1. choose samples in probability p
    2. random set k head to zero in selected samples
    """

    def __init__(self, p: float=0.5, head_drop_upper: int=1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                'dropout probability has to be between 0 and 1, but got {}'
                .format(p))
        self.p = p
        self.upper = head_drop_upper

    def forward(self, x):
        B, N, c, num, dim = x.shape
        if self.training and self.p > 0 and self.upper > 0:
            batch_rand = paddle.rand((B,))
            batch_mask = (batch_rand <= self.p).numpy()
            all_indexs = [i for i in range(num)]
            weights = [None] * B
            for i in range(B):
                if batch_mask[i]:
                    k = random.randint(1, self.upper)
                    zero_index = random.choices(all_indexs, k=k, weights=\
                        weights[i])
                    x[i, zero_index, ...] = 0
        return x


class Attention(nn.Layer):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
        attn_drop=0.0, proj_drop=0.0, head_drop_type='Dropout',
        head_drop_rate=0.0, head_drop_upper=1):
        super().__init__()
        assert head_drop_type in (None, 'Dropout', 'HeadDropOut')
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        if head_drop_type == 'Dropout':
            self.head_drop = nn.Dropout(head_drop_rate)
        elif head_drop_type == 'HeadDropOut':
            self.head_drop = HeadDropOut(head_drop_rate, head_drop_upper=\
                head_drop_upper)
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
        .GELU, norm_layer=nn.LayerNorm, head_drop_type='Dropout',
        head_drop_rate=0.0, head_drop_upper=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            head_drop_type=head_drop_type, head_drop_rate=head_drop_rate,
            head_drop_upper=head_drop_upper)
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


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=\
        768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, head_drop_type=\
        'HeadDropOut', head_drop_rate=0.0, head_drop_upper=1,
        drop_path_rate=0.0, hybrid_backbone=None, norm_layer_eps=1e-05,
        patch_stride=None, patch_padding=0, freeze=False, input_type=\
        'image', num_patches=None, pretrained=None, **kwargs):
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
        self.pos_embed = self.create_parameter(shape=(1, num_patches + 1, embed_dim))
        self.add_parameter("pos_embed", self.pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                head_drop_type=head_drop_type,
                head_drop_rate=head_drop_rate,
                head_drop_upper=head_drop_upper
            )for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, paddle.nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, paddle.nn.Linear) and m.bias is not None:
    #             torch2paddle.constant_init_(m.bias, 0)
    #     elif isinstance(m, paddle.nn.LayerNorm):
    #         torch2paddle.constant_init_(m.bias, 0)
    #         torch2paddle.constant_init_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        if self.input_type == 'image':
            x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        # x = torch2paddle.concat((cls_tokens, x), dim=1)
        x = paddle.concat((cls_tokens, x), axis=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
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


class TransFER(nn.Layer):
    """TransFER in ICCV"""

    def __init__(self, class_num=7):
        super().__init__()
        self.extractor = IRSE(input_size=(112, 112), num_layers=44, mode='ir')
        self.attention = MultiLANet(in_channels=256, ratio=16, num=2,
            loss_margin=1.0, loss_weight=0.0)
        self.convert = MaxHybridFlatten(input_shape=(14, 14, 256),
            embed_dim=768, drop_rate=0.6, drop_upper=1, adaptive_drop=False)
        self.vit = VisionTransformer(input_type='feature', num_patches=196,
            head_drop_type='HeadDropOut', head_drop_rate=0.3,
            head_drop_upper=1, patch_size=16, embed_dim=768, depth=8,
            num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-06)
        self.head = LinearClsHead(num_classes=class_num, in_channels=768)

    def forward(self, img, gt_label=None):
        x = self.extractor(img)
        x = self.attention(x)
        x = self.convert(img=img, **x)
        x = self.vit(x)
        x = self.head(x, gt_label)
        return x


if __name__ == '__main__':
    model = TransFER()
    ckpt = paddle.load(
        '../RAFa_2_6131_ep33_90p91_param.pdiparams')
    for k in list(ckpt.keys()):
        if k.split('.')[-1] == 'weight':
            new_key = k[:-6] + '_weight'
            ckpt[new_key] = ckpt[k]
    # print(ckpt.keys())
    model.load_state_dict(ckpt)
    model.eval()
    mock_input = paddle.rand(1, 3, 112, 112)
    with paddle.no_grad():
        out = model(mock_input)
    print(out.shape)
