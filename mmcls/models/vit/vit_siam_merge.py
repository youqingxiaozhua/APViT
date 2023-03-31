"""
input: 
    stage2 and stage3 tokens from CNN

forward:
    s2 and s3 processed by 3 MHSA blocks, seperately
    the output are concatenated and feed to 3+2 MHSA
    add a skip connection to last 2 MHSA

"""
import math
import os
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from functools import partial
from typing import List

from mmcls.utils import get_root_logger
from mmcv.runner import load_state_dict
from mmcls.models.necks.la_max import LANet
from ..builder import BACKBONES, build_loss
from ..backbones.base_backbone import BaseBackbone
from .layers import DropPath, to_2tuple, trunc_normal_, resize_pos_embed_v2
from ..utils import top_pool
from ..backbones.modules.vit import Block
from ..backbones.modules.vit_pooling import PoolingBlock


@BACKBONES.register_module()
class CNNMultiStageCatOrigin(BaseBackbone):
    """
    CNN 不同 stage 的 Concate 在一起
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_nums=[],
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature'
        if input_type == 'feature':
            assert len(in_channels) == len(patch_nums), 'Input channels must have the same length of patch_nums'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, sum(patch_nums), embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                )
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)


        if pretrained:
            self.init_weights(pretrained, patch_nums)
        else:
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_nums=[]):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]
        pos_embeds = []
        for patch_num in patch_nums:
            print('patch_num', patch_num)
            if patch_num != pos_embed.shape[1] - 1:
                logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
                pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
            else:   # 去掉 cls_token
                print('does not need to resize')
                pos_embed_new = patch_pos_embed
            pos_embeds.append(pos_embed_new)
        state_dict['pos_embed'] = torch.cat(pos_embeds, dim=1)
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)
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

    def forward_features(self, x):
        x = [self.projs[i](x[i]) for i in range(len(x))]
        x = [i.flatten(2).transpose(2, 1) for i in x]

        x = torch.cat(x, dim=1)
        
        B = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)
        x_pos = torch.cat((self.cls_pos_embed, self.pos_embed), dim=1)
        x = x + x_pos
   
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)    # (B, N, dim)
        if self.training is False:
            print(x.shape)
        loss = dict()
        return x[:, 0], loss


    def forward(self, x, **kwargs):
        x, loss = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss))


@BACKBONES.register_module()
class CNNMultiStageIntraAdd(BaseBackbone):
    """ 
    CNN 不同 stage 的 feature 经过 siam_block 来探索 stage 内的关系，
    然后 add 在一起探索 stage 间的关系
    Intra(S2) + Intra(S3)
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 s2_pooling_method='MaxPooling',
                 siam_depth=1,
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                )
            for i in range(depth)])
        self.siam_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                )
            for i in range(siam_depth)])
        self.norm = norm_layer(embed_dim)
        if s2_pooling_method == 'MaxPooling':
            self.s2_pooling = nn.MaxPool2d(kernel_size=2)
        elif s2_pooling_method == 'Unfold':
            self.s2_pooling = nn.Unfold(kernel_size=2, stride=2)    # Unfold instead of MaxPooling
        else:
            raise ValueError("Unknown pooling method: %s" % s2_pooling_method)


        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # 去掉 cls_token
            print('does not need to resize')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        # copy weights from blocks to siam blocks
        siam_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith(f'blocks.'):
                if int(k.split('.')[1]) in list(range(len(self.siam_blocks))):
                    siam_state_dict[k.replace('blocks.', 'siam_blocks.')] = v
        state_dict.update(siam_state_dict)

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

    def forward_features(self, x):
        assert len(x) == 2, '目前只支持2个 stage'
        B, C, H, W = x[-1].shape
        x[0] = self.s2_pooling(x[0]).reshape(B, -1, H, W)

        x2D = [self.projs[i](x[i]) for i in range(len(x))]
        x = [i.flatten(2).transpose(2, 1) for i in x2D]
        if len(self.siam_blocks) == 0:
            x = [x[i] + self.patch_pos_embed/len(x) for i in range(len(x))]
        else:
            x = [x[i] + self.patch_pos_embed for i in range(len(x))]

        x0, x1 = x[0], x[1]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if len(self.siam_blocks) == 0:
            cls_tokens = cls_tokens + self.cls_pos_embed/len(x)
        else:
            cls_tokens = cls_tokens + self.cls_pos_embed

        x0 = torch.cat((cls_tokens, x0), dim=1)
        x1 = torch.cat((cls_tokens, x1), dim=1)

        x0 = self.pos_drop(x0)
        x1 = self.pos_drop(x1)

        # Intra
        for blk in self.siam_blocks:
            x0 = blk(x0)

        for blk in self.blocks[:len(self.siam_blocks)]:
            x1 = blk(x1)

        x = x0 + x1

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)    # (B, N, dim)
        if self.training is False:
            print(x.shape)
        loss = dict()
        return x[:, 0], loss


    def forward(self, x, **kwargs):
        x, loss = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss))


@BACKBONES.register_module()
class PoolingViT(BaseBackbone):
    """ 
    
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='SUM_ABS_1',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 multi_head_fusion=False,
                 sum_batch_mean=False,
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature', 'Only suit for hybrid model'
        self.sum_batch_mean = sum_batch_mean
        if sum_batch_mean:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.multi_head_fusion = multi_head_fusion
        self.num_heads = num_heads
        if multi_head_fusion:
            assert vit_pool_configs is None, 'MultiHeadFusion only support original ViT Block, by now'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            # self.attn_f = LANet(in_channels[-1], 16)
            self.attn_f = LANet(embed_dim, 16)
            
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        elif attn_method == 'Random':
            self.attn_f = lambda x: x[:, torch.randint(high=x.shape[1], size=(1,))[0], ...].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:

            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=multi_head_fusion,
                    )
                for i in range(depth)])
        else:
            vit_keep_rates = vit_pool_configs['keep_rates']
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_keep_rates[i], **vit_pool_configs),
                    )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # remove cls_token
            print('does not need to resize!')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        if self.multi_head_fusion:
            # convert blocks.0.attn.qkv.weight to blocks.0.attn.qkv.0.weight
            num_groups = self.blocks[0].attn.group_number
            d = self.embed_dim // num_groups
            print('d', d)
            for k in list(state_dict.keys()):
                if k.startswith('blocks.'):
                    keys = k.split('.')
                    if  not (keys[2] == 'attn' and keys[3] == 'qkv'):
                        continue
                    for i in range(num_groups):
                        new_key = f'blocks.{keys[1]}.attn.qkv.{i}.weight'
                        new_value = state_dict[k][i*3*d:(i+1)*3*d, i*d: i*d+d]
                        state_dict[new_key] = new_value

                    del state_dict[k]

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
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

    def forward_features(self, x):
        assert len(x) == 1, '目前只支持1个 stage'
        assert isinstance(x, list) or isinstance(x, tuple)
        if len(x) == 2: # S2, S3
            x[0] = self.s2_pooling(x[0])
        elif len(x) == 3:
            x[0] = nn.MaxPool2d(kernel_size=4)(x[0])
            x[1] = self.s2_pooling(x[1])
        if os.getenv('DEBUG_MODE') == '1':
            print(x[0].shape)

        x = [self.projs[i](x[i]) for i in range(len(x))]
        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient
        x = [i.flatten(2).transpose(2, 1) for i in x]
        # x = self.projs[0](x).flatten(2).transpose(2, 1)
        # disable the first row and columns
        # attn_map[:, :, 0, :] = 0.
        # attn_map[:, :, :, 0] = 0.
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        # attn_weight = torch.rand(attn_weight.shape, device=attn_weight.device)
        
        x = torch.stack(x).sum(dim=0)   # S1 + S2 + S3
        x = x + self.patch_pos_embed
        
        B, N, C = x.shape
        
        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            if keep_indexes is not None:
                x = x.gather(dim=1, index=keep_indexes)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)    # (B, N, dim)
        if os.environ.get('DEBUG_MODE', '0') == '1':
            print('output', x.shape)
        x = x[:, 0]
        if self.sum_batch_mean:
            x = x + x.mean(dim=0) * self.alpha
        loss = dict()
        return x, loss, attn_map

    def forward(self, x, **kwargs):
        x, loss, attn_map = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss), attn_map=attn_map)


@BACKBONES.register_module()
class HeadFusionViT(BaseBackbone):
    """ 
    
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='LA',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 multi_head_fusion=False,
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature'
        self.multi_head_fusion = multi_head_fusion
        self.num_heads = num_heads
        if multi_head_fusion:
            assert vit_pool_configs is None, 'MultiHeadFusion only support original ViT Block, by now'

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            self.attn_f = LANet(in_channels[-1], 16)
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=multi_head_fusion,
                    )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
            PoolingBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                pool_config=dict(keep_rate=vit_pool_configs['keep_rates'][i], **vit_pool_configs),
                )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)


    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # 去掉 cls_token
            print('does not need to resize')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        if self.multi_head_fusion:
            # convert blocks.0.attn.qkv.weight to blocks.0.attn.qkv.0.weight
            num_groups = self.blocks[0].attn.group_number
            d = self.embed_dim // num_groups
            print('d', d)
            for k in list(state_dict.keys()):
                if k.startswith('blocks.'):
                    keys = k.split('.')
                    if  not (keys[2] == 'attn' and keys[3] == 'qkv'):
                        continue
                    for i in range(num_groups):
                        new_key = f'blocks.{keys[1]}.attn.qkv2.{i}.weight'
                        new_value = state_dict[k][i*3*d:(i+1)*3*d, i*d: i*d+d]
                        state_dict[new_key] = new_value

                    # del state_dict[k]

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
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

    def forward_features(self, x):
        assert len(x) == 1, '目前只支持1个 stage'
        x = x[0]
        B, C, H, W = x.shape
        attn_map = self.attn_f(x) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x = x * attn_map    #  to have gradient
        x = self.projs[0](x).flatten(2).transpose(2, 1)
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        x = x + self.patch_pos_embed
        
        B, N, C = x.shape
        
        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            if keep_indexes is not None:
                x = x.gather(dim=1, index=keep_indexes)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)    # (B, N, dim)
        if os.environ.get('DEBUG_MODE', '0') == '1':
            print(x.shape)
        loss = dict()
        return x[:, 0], loss

    def forward(self, x, **kwargs):
        x, loss = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss))


@BACKBONES.register_module()
class DeFPNViTV2(BaseBackbone):
    """ 
    浅层feature经过更多的block，高层经过的block少
    CNN每个stage再经过个Pooling
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='LA',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 insert_index=(0, 1, 2),
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature', 'Only suit for hybrid model'
        assert len(insert_index) == len(in_channels), 'insert_index must have the same length as in_channels'
        assert max(insert_index) < depth, 'insert_index must be smaller than depth'
        self.insert_index = insert_index
        self.num_heads = num_heads
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        # self.patch_pos_embed_0 = nn.Parameter(torch.zeros(1, patch_num * 16, embed_dim), requires_grad=True)
        self.patch_pos_embed_1 = nn.Parameter(torch.zeros(1, patch_num * 4, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            self.attn_f = LANet(in_channels[-1], 16)
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=False,
                    )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_pool_configs['keep_rates'][i], **vit_pool_configs),
                    )
            for i in range(depth)])
        # self.norm = norm_layer(embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim, momentum=0.9, affine=False)

        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

        # self.test_sum = torch.load('output/sum.pth')

    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # 去掉 cls_token
            print('does not need to resize')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new
        # without maxpooling
        # pos_embed_new_0 = resize_pos_embed_v2(patch_pos_embed, patch_num*16, 0)
        # state_dict['patch_pos_embed_0'] = pos_embed_new_0
        pos_embed_new_1 = resize_pos_embed_v2(patch_pos_embed, patch_num*4, 0)
        state_dict['patch_pos_embed_1'] = pos_embed_new_1

        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
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

    def forward_features(self, x):
        assert isinstance(x, list)
        # if len(x) == 2: # S2, S3
        #     x[0] = self.s2_pooling(x[0])
        # elif len(x) == 3:
        #     x[0] = nn.MaxPool2d(kernel_size=4)(x[0])
        #     x[1] = self.s2_pooling(x[1])
        
        x = [self.projs[i](x[i]) for i in range(len(x))]


        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient

        attn_weights = [self.attn_f(i).flatten(2).transpose(2, 1) for i in x]
        x = [i.flatten(2).transpose(2, 1) for i in x]


        # x = [i + self.patch_pos_embed for i in x]
        pos_embes = [self.patch_pos_embed_1, self.patch_pos_embed]
        x = [x[i] + pos_embes[i] for i in range(len(x))]

        if self.cnn_pool_config is not None:
            # keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            keep_indexes = [
                top_pool(
                    attn_weights[i], 
                    dim=C, keep_num=self.cnn_pool_config['keep_num'][i], exclude_first=False
                ) for i in range(len(x))]
            x = [x[i].gather(dim=1, index=keep_indexes[i]) if keep_indexes[i] is not None else x[i] for i in range(len(x))]

        # x = x + self.patch_pos_embed
        
        # B, N, C = x.shape
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        for i, blk in enumerate(self.blocks):
            if i == self.insert_index[0]:
                flow = torch.cat((cls_tokens, x[0]), dim=1)
            elif i == self.insert_index[1]:
                flow[:, 1:, :] = x[1] + flow[:, 1:, :]  # type: ignore  enforce flow to define in index[0],
            # elif i == self.insert_index[2]:
            #     flow[:, 1:, :] = x[2] + flow[:, 1:, :]  # type: ignore
            flow = blk(flow)    # type: ignore

        # for blk in self.blocks:
        #     x = blk(x)
        flow = flow[:, 0]
        x_static = self.norm(flow)    # type: ignore (B, N, dim)
        x = flow
        if os.environ.get('DEBUG_MODE', '0') == '1':
            print(x.shape)
        loss = dict()
        # if self.batch_mean is None:
        #     self.batch_mean =x.mean(dim=0)
        # x = x + self.batch_mean * self.alpha
        # if self.test_sum is None:
        #     self.test_sum = x.sum(dim=0)
        # else:
        #     self.test_sum += x.sum(dim=0)
        # torch.save(self.test_sum, 'output/sum.pth')
        # x = x + x.mean(dim=0) * self.alpha
        x = x + self.norm.running_mean * self.alpha
        # print(self.alpha)
        # x = x[:, 0]
        # x = x + self.norm.running_mean * self.alpha
        return x, loss
        # return x[:, 0], loss

    def forward(self, x, **kwargs):
        x, loss = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss))


@BACKBONES.register_module()
class DeFPNViTV5(BaseBackbone):
    """ 
    浅层feature经过更多的block，高层经过的block少
    CNN每个stage再经过个Pooling
    S2的pooling追随S3，ViT中前期不pooling，只相加
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='LA',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 insert_index=(0, 1, 2),
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature', 'Only suit for hybrid model'
        assert len(insert_index) == len(in_channels), 'insert_index must have the same length as in_channels'
        assert max(insert_index) < depth, 'insert_index must be smaller than depth'
        self.insert_index = insert_index
        self.num_heads = num_heads
        # self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        # self.patch_pos_embed_0 = nn.Parameter(torch.zeros(1, patch_num * 16, embed_dim), requires_grad=True)
        # self.patch_pos_embed_1 = nn.Parameter(torch.zeros(1, patch_num * 4, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            self.attn_f = LANet(in_channels[-1], 16)
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=False,
                    )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_pool_configs['keep_rates'][i], **vit_pool_configs),
                    )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # self.norm = nn.BatchNorm1d(embed_dim, momentum=0.9, affine=False)

        self.s1_pooling = nn.MaxPool2d(kernel_size=4)
        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # 去掉 cls_token
            print('does not need to resize')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new

        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
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

    def forward_features(self, x):
        assert isinstance(x, list)
        assert len(x) == 2, 'The SOTA of proposed model should use S2 and S3 ( \
            MultiHeadFusion or DeFPN)'
        if len(x) == 2: # S2, S3
            x[0] = self.s2_pooling(x[0])
        elif len(x) == 3:
            x[0] = self.s1_pooling(x[0])
            x[1] = self.s2_pooling(x[1])
        
        x = [self.projs[i](x[i]) for i in range(len(x))]


        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient
        
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        x = [i.flatten(2).transpose(2, 1) for i in x]


        x = [i + self.patch_pos_embed for i in x]

        if self.cnn_pool_config is not None:
            keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            x = [x[i].gather(dim=1, index=keep_indexes) if keep_indexes is not None else x[i] for i in range(len(x))]

        # x = x + self.patch_pos_embed
        
        # B, N, C = x.shape
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        for i, blk in enumerate(self.blocks):
            if i == self.insert_index[0]:
                flow = torch.cat((cls_tokens, x[0]), dim=1)
            elif i == self.insert_index[1]:
                flow[:, 1:, :] = x[1] + flow[:, 1:, :]  # type: ignore  enforce flow to define in index[0],
            # elif i == self.insert_index[2]:
            #     flow[:, 1:, :] = x[2] + flow[:, 1:, :]  # type: ignore
            flow = blk(flow)    # type: ignore

        flow = self.norm(flow)

        if os.environ.get('DEBUG_MODE', '0') == '1':
            print(flow.shape)
            # exit()
        x = flow[:, 0]
        loss = dict()
        return x, loss
        # return x[:, 0], loss

    def forward(self, x, **kwargs):
        x, loss = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss))



@BACKBONES.register_module()
class DeFPNViTV7(BaseBackbone):
    """ 
    浅层feature经过更多的block，高层经过的block少
    CNN每个stage再经过个Pooling
    S2的pooling追随S3，但keep num不同
    ViT前期就pooling
    """
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., norm_layer_eps=1e-5, freeze=False,
                 input_type='image',  pretrained=None, 
                 in_channels=[], patch_num=0,
                 attn_method='SUM_ABS_1',
                 cnn_pool_config=None,
                 vit_pool_configs=None,
                 insert_index=(0, 1, 2),
                 **kwargs):
        super().__init__()
        if kwargs:
            print('Unused kwargs: ')
            print(kwargs)
        assert input_type  == 'feature', 'Only suit for hybrid model'
        assert len(insert_index) == len(in_channels), 'insert_index must have the same length as in_channels'
        assert max(insert_index) < depth, 'insert_index must be smaller than depth'
        self.insert_index = insert_index
        self.num_heads = num_heads
        # self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        self.input_type = input_type
        norm_layer = partial(nn.LayerNorm, eps=norm_layer_eps)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.projs = nn.ModuleList([nn.Conv2d(in_channels[i], embed_dim, 1,) for i in range(len(in_channels))])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.patch_pos_embed = nn.Parameter(torch.zeros(1, patch_num, embed_dim), requires_grad=True)
        # self.patch_pos_embed_0 = nn.Parameter(torch.zeros(1, patch_num * 16, embed_dim), requires_grad=True)
        # self.patch_pos_embed_1 = nn.Parameter(torch.zeros(1, patch_num * 4, embed_dim), requires_grad=True)
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attn_method = attn_method
        self.cnn_pool_config = cnn_pool_config
        if attn_method == 'LA':
            self.attn_f = LANet(in_channels[-1], 16)
        elif attn_method == 'SUM':
            self.attn_f = lambda x: torch.sum(x, dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_1':
            self.attn_f = lambda x: torch.sum(torch.abs(x), dim=1).unsqueeze(1)
        elif attn_method == 'SUM_ABS_2':
            self.attn_f = lambda x: torch.sum(torch.pow(torch.abs(x), 2), dim=1).unsqueeze(1)
        elif attn_method == 'MAX':
            self.attn_f = lambda x: torch.max(x, dim=1)[0].unsqueeze(1)
        elif attn_method == 'MAX_ABS_1':
            self.attn_f = lambda x: torch.max(torch.abs(x), dim=1)[0].unsqueeze(1)
        else:
            raise ValueError("Unknown attn_method")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if vit_pool_configs is None:
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, head_fusion=False,
                    )
                for i in range(depth)])
        else:
            assert len(vit_pool_configs['keep_rates']) == depth, 'Keep rates must have the same length of depth'
            self.blocks = nn.ModuleList([
                PoolingBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                    pool_config=dict(keep_rate=vit_pool_configs['keep_rates'][i], **vit_pool_configs),
                    )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # self.norm = nn.BatchNorm1d(embed_dim, momentum=0.9, affine=False)

        self.s1_pooling = nn.MaxPool2d(kernel_size=4)
        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        if pretrained:
            self.init_weights(pretrained, patch_num)
        else:
            trunc_normal_(self.patch_pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            self.apply(self._init_weights)
        if freeze:
            self.apply(self._freeze_weights)

    def init_weights(self, pretrained, patch_num=0):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed']     # [1, 197, 768] for base
        patch_pos_embed = pos_embed[:, 1:, :]

        if patch_num != pos_embed.shape[1] - 1:
            logger.warning(f'interpolate pos_embed from {patch_pos_embed.shape[1]} to {patch_num}')
            pos_embed_new = resize_pos_embed_v2(patch_pos_embed, patch_num, 0)
        else:   # 去掉 cls_token
            print('does not need to resize')
            pos_embed_new = patch_pos_embed
        del state_dict['pos_embed']
        state_dict['patch_pos_embed'] = pos_embed_new

        state_dict['cls_pos_embed'] = pos_embed[:, 0, :].unsqueeze(1)

        for k in ('patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias'):
            del state_dict[k]
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

    def forward_features(self, x):
        assert isinstance(x, list)
        if len(x) == 2: # S2, S3
            x[0] = self.s2_pooling(x[0])
        elif len(x) == 3:
            x[0] = self.s1_pooling(x[0])
            x[1] = self.s2_pooling(x[1])
        
        x = [self.projs[i](x[i]) for i in range(len(x))]


        # x = x[0]
        B, C, H, W = x[-1].shape
        attn_map = self.attn_f(x[-1]) # [B, 1, H, W]
        if self.attn_method == 'LA':
            x[-1] = x[-1] * attn_map    #  to have gradient
        
        attn_weight = attn_map.flatten(2).transpose(2, 1)

        x = [i.flatten(2).transpose(2, 1) for i in x]


        x = [i + self.patch_pos_embed for i in x]

        if self.cnn_pool_config is not None:
            assert len(self.cnn_pool_config['keep_num']) == len(x), 'x must have the same length of keep_num'
            # keep_indexes = top_pool(attn_weight, dim=C, **self.cnn_pool_config)
            keep_indexes = [top_pool(attn_weight, dim=C, keep_num=self.cnn_pool_config['keep_num'][i], exclude_first=False) for i in range(len(x))]
            x = [x[i].gather(dim=1, index=keep_indexes[i]) if keep_indexes[i] is not None else x[i] for i in range(len(x))]

        # x = x + self.patch_pos_embed
        
        # B, N, C = x.shape
        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        cls_tokens = cls_tokens + self.cls_pos_embed

        for i, blk in enumerate(self.blocks):
            if i == self.insert_index[0]:
                flow = torch.cat((cls_tokens, x[0]), dim=1)
            elif i == self.insert_index[1]:
                flow[:, 1:, :] = x[1] + flow[:, 1:, :]  # type: ignore  enforce flow to define in index[0],
            # elif i == self.insert_index[2]:
            #     flow[:, 1:, :] = x[2] + flow[:, 1:, :]  # type: ignore
            flow = blk(flow)    # type: ignore

        # for blk in self.blocks:
        #     x = blk(x)
        flow = self.norm(flow)
        if os.environ.get('DEBUG_MODE', '0') == '1':
            print(flow.shape)
            exit()
        x = flow[:, 0]

        loss = dict()
        return x, loss
        # return x[:, 0], loss

    def forward(self, x, **kwargs):
        x, loss = self.forward_features(x)
        return dict(x=x, loss=dict(VitDiv_loss=loss))


