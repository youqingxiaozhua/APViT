""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""
import math
import random
import logging
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from functools import partial

from mmcv.runner import load_state_dict
from ..builder import BACKBONES
from ..backbones.base_backbone import BaseBackbone
from .layers import DropPath, to_2tuple, trunc_normal_, resize_pos_embed


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        # 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
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


class HeadDropOut(nn.Module):
    """
    1. choose samples in probability p
    2. random set k head to zero in selected samples
    """
    def __init__(self, p: float = 0.5, head_drop_upper: int=1, adaptive_drop=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.upper = head_drop_upper
        self.adaptive_drop = adaptive_drop
        if adaptive_drop:
            print('ViT HeadDropOut Adaptive Drop enabled')
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, c, num, dim = x.shape # c=3, # 8 197 3 8 96
        if self.training and self.p > 0 and self.upper > 0:
            batch_rand = torch.rand((B, ))
            batch_mask = batch_rand <= self.p
            # random set k head to zero
            all_indexs = [i for i in range(num)]
            if self.adaptive_drop:
                weight_feature = x.reshape((B, N*c, num, dim)).permute((0, 2, 1, 3))
                # print(B, N, c, num, dim)
                # print(weight_feature.shape,)
                weights = nn.AdaptiveAvgPool2d(1)(weight_feature).reshape(B, num)
                weights += weights.min() + 1
                weights = torch.add(torch.sqrt(weights), 1e-12)
                weights = torch.div(weights, torch.sum(weights,dim=1).unsqueeze(1))
            #     weights = weights.cpu().numpy()
            else:
                weights = [None] * B

            for i in range(B):
                if batch_mask[i]:
                    k = random.randint(1, self.upper)
                    zero_index = random.choices(all_indexs, k=k, weights=weights[i])
                    x[i, zero_index, ...] = 0
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., 
                 head_drop_type='Dropout', head_drop_rate=0., head_drop_upper=1):
        super().__init__()
        assert head_drop_type in (None, 'Dropout', 'HeadDropOut')
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if head_drop_type == 'Dropout':
            self.head_drop = nn.Dropout(head_drop_rate)
        elif head_drop_type == 'HeadDropOut':
            self.head_drop = HeadDropOut(head_drop_rate, head_drop_upper=head_drop_upper)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.center = None

    # def calculate_center_loss(self, features, center):
    #     """
    #     args:
    #         features: [B, Heads, Tokens, Tokens]
    #         center: [H, TxT]
    #     output:
    #         center_loss: scalar
    #         diff: [H, TxT]
    #     """
    #     # features = features.clone().detach()
    #     features = features.reshape(features.shape[0], features.shape[1], -1)

    #     # center = torch.nn.functional.normalize(center, dim=-1)
    #     # features = torch.nn.functional.normalize(features, dim=-1)
    #     diff = features - center     # [B, H, TxT]
    #     distance = torch.pow(diff, 2)
    #     distance = torch.sqrt(distance)
    #     center_loss = torch.mean(distance)

    #     batch_center = torch.mean(features.detach(), dim=0)
    #     return center_loss, batch_center

    def calculate_center_loss(self, features, center):
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.mean(dim=1)  # [B, c]
        diff =  features.detach() - center
        distance = torch.pow(features - center, 2)
        center_loss = torch.mean(distance)
        diff = torch.mean(diff, dim=0) # mean for batch
        return center_loss, diff

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        if hasattr(self, 'head_drop'):
            qkv = self.head_drop(qkv)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, head_num, token_num, token_num]

        first_step = False
        if self.center is None:
            first_step = True
            self.center = attn.clone().detach().mean(dim=1).mean(dim=0).flatten(start_dim=-2)
        center_loss, batch_center = self.calculate_center_loss(attn, self.center)
        momentum = 0.8
        loss_weight = 0.
        center_loss *= loss_weight
        if first_step:
            self.center = batch_center
        else:
            self.center = self.center * momentum + batch_center * (1-momentum)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, center_loss


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, head_drop_type='Dropout', head_drop_rate=0., head_drop_upper=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            head_drop_type=head_drop_type, head_drop_rate=head_drop_rate, head_drop_upper=head_drop_upper)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # self.center = torch.zeros(dim, device='cuda')
        self.center = None
    
    def calculate_pooling_center_loss(self, features, center, alpha=0.95):
        """
        args:
            features: [B, N, 768], N: token number
            center: [N, 768]
        output:
            center_loss: scalar
            diff: [N, 768]
        """
        # features = features.reshape(features.shape[0], -1)
        centers_batch = torch.nn.functional.normalize(center, dim=-1)
        features = features.clone().detach()
        diff =  (1-alpha)*(features - centers_batch)
        distance = torch.pow(features - centers_batch, 2)
        distance = torch.sum(distance, dim=-1)
        center_loss = torch.mean(distance)
        diff = torch.mean(diff, dim=0)
        return center_loss, diff

    def forward(self, x):
        feature, center_loss = self.attn(self.norm1(x))
        # B, N, dim = feature.shape
        # if self.center is None:
        #     self.center = torch.zeros(feature.shape[1:], device='cuda')
        # center_loss, diff = self.calculate_pooling_center_loss(feature, self.center)
        # center_loss *= 0
        # self.center += diff
        # center_loss = 0
        x = x + self.drop_path(feature)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, center_loss


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None, padding=0):
        super().__init__()
        if stride is None:
            stride = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        num_patches = (img_size[0] + 2 * padding - patch_size[0]) / stride + 1
        if num_patches % 1 != 0:
            print('Warning: patch numbser is not int:', num_patches)
        num_patches = int(num_patches) ** 2
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)


    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
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


@BACKBONES.register_module()
class VisionTransformer(BaseBackbone):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 head_drop_type='HeadDropOut', head_drop_rate=0., head_drop_upper=1,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer_eps=1e-5, patch_stride=None, patch_padding=0, freeze=False,
                 input_type='image', num_patches=None, pretrained=None, **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type in ('image', 'feature'), 'input_type must be "image" or "feature"'
        if input_type == 'feature':
            assert num_patches is not None
        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)
        # self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if input_type == 'image':
            if hybrid_backbone is not None:
                self.patch_embed = HybridEmbed(
                    hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_stride, padding=patch_padding)
            num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                head_drop_type=head_drop_type, head_drop_rate=head_drop_rate, head_drop_upper=head_drop_upper)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        if pretrained:
            self.init_weights(pretrained, num_patches + 1)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=None):
        assert isinstance(patch_num, int)
        logger = logging.getLogger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        if patch_num is not None and patch_num != pos_embed.shape[1]:
            logger.warning(f'interpolate pos_embed from {pos_embed.shape[1]} to {patch_num}')
            pos_embed = resize_pos_embed(pos_embed, self.pos_embed, 1)
        state_dict['pos_embed'] = pos_embed
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _freeze_weights(self, m):
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
        for param in m.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        if self.input_type == 'image':
            x = self.patch_embed(x)  # (B, N, C) N: patch number

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        center_loss_sum = 0

        for blk in self.blocks:
            x, center_loss = blk(x)
            center_loss_sum = center_loss

        # center_loss = center_loss_sum / len(self.blocks)
        center_loss = center_loss_sum   # only compute the last block

        x = self.norm(x)    # (B, N, dim)
        return x[:, 0], center_loss
        # x = torch.mean(x, dim=1)    # global_avgpool
        # return x, center_loss

    def forward(self, x):
        x, center_loss = self.forward_features(x)
        # x = self.head(x)
        return x, center_loss


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def vit_small_patch16_224(pretrained=False, **kwargs):
    if pretrained:
        # NOTE my scale was wrong for original weights, leaving this here until I have better ones for this model
        kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_base_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_base_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch16_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_large_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,  qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_cfgs['vit_large_patch32_384']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def vit_huge_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(patch_size=16, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch16_224']
    return model


def vit_huge_patch32_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=32, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, **kwargs)
    model.default_cfg = default_cfgs['vit_huge_patch32_384']
    return model


def vit_small_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet26d_224']
    return model


def vit_small_resnet50d_s3_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[3])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_small_resnet50d_s3_224']
    return model


def vit_base_resnet26d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet26d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet26d_224']
    return model


def vit_base_resnet50d_224(pretrained=False, **kwargs):
    pretrained_backbone = kwargs.get('pretrained_backbone', True)  # default to True for now, for testing
    backbone = resnet50d(pretrained=pretrained_backbone, features_only=True, out_indices=[4])
    model = VisionTransformer(
        img_size=224, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, hybrid_backbone=backbone, **kwargs)
    model.default_cfg = default_cfgs['vit_base_resnet50d_224']
    return model