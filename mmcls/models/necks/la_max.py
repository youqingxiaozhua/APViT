# use for Max(multi LA) -> HybridEmbd

import random
from typing import List
import torch
from torch import nn, Tensor
from mmcv.runner import load_state_dict
# from mmcls.utils import get_root_logger

from ..builder import NECKS, build_loss
from ..utils import top_pool
from ..vit.layers import resize_pos_embed, trunc_normal_


class LANet(nn.Module):
    def __init__(self, channel_num, ratio=16):
        super().__init__()
        assert channel_num % ratio == 0, f"input_channel{channel_num} must be exact division by ratio{ratio}"
        self.channel_num = channel_num
        self.ratio = ratio
        self.relu = nn.ReLU(inplace=True)

        self.LA_conv1 = nn.Conv2d(channel_num, int(channel_num / ratio), kernel_size=1)
        self.bn1 = nn.BatchNorm2d(int(channel_num / ratio))
        self.LA_conv2 = nn.Conv2d(int(channel_num / ratio), 1, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        LA = self.LA_conv1(x)
        LA = self.bn1(LA)
        LA = self.relu(LA)
        LA = self.LA_conv2(LA)
        LA = self.bn2(LA)
        LA = self.sigmoid(LA)
        return LA
        # LA = LA.repeat(1, self.channel_num, 1, 1)
        # x = x*LA

        # return x


@NECKS.register_module()
class MLPConvert(nn.Module):
    """
    One layer MLP to project CNN feature to ViT token
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert len(in_channels) == len(out_channels)
        self.proj = nn.ModuleList([nn.Conv2d(in_channels[i], out_channels[i], 1,) for i in range(len(in_channels))])
    
    def forward(self, x:List[Tensor]):
        assert isinstance(x, (list, tuple))
        out = [self.proj[i](x[i]) for i in range(len(x))]
        out = [i.flatten(2).transpose(2, 1) for i in out]
        return dict(x=out)


@NECKS.register_module()
class MultiLANet(nn.Module):
    """Multi LANet.
    from paper: Hierarchical pyramid diverse attention networks for face recognition
    
    return: [Tensor(B,1,H,W), ], loss
    """

    def __init__(self, in_channels, ratio=16, num=5, 
        loss=dict(type='LANetAttention_div_loss', loss_margin=1.0, loss_weight=0.5)
        ):
        super().__init__()
        # assert num >=2, 'only 1 LANet can not compute loss'
        self.num = num
        self.criterion = build_loss(loss)
        self.las =  nn.ModuleList([LANet(in_channels, ratio) for _ in range(num)])

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            raise TypeError('tuple inputs are not suported by MultiLANet now')
        elif isinstance(inputs, torch.Tensor):
            if self.num == 0:
                return dict(feature=inputs, la_outs=None, loss=dict(LADiv_loss=torch.as_tensor([0.,], device='cuda')))
            la_outs = [f(inputs) for f in self.las]
            loss = self.criterion(la_outs)
        else:
            raise TypeError(f'neck inputs should be tuple or torch.tensor, but get {type(inputs)}')
        return dict(feature=inputs, la_outs=la_outs, loss=dict(LADiv_loss=loss))


@NECKS.register_module()
class ConvFlatten(nn.Module):
    """
    use Conv2D to flatten (C,H,W)
    input:
        input_shape: h, w, c
        x: [ Tensor((B, 1, H, W)), ]
    
    process:
        [reshape_mlp]
        [bilinear]
        mlp

    output: Tensor((B, n, c)), n = len(x)
    """
    def __init__(self, input_shape, embed_dim, resize_shape=None, resize_method='bilinear', pretrained=None):
        super().__init__()
        assert len(input_shape) == 3, "input_shape must have 3 dimensions"
        assert resize_method in ('bilinear', 'padding')
        self.resize_shape = resize_shape
        self.resize_method = resize_method

        if resize_shape and resize_method == 'padding':
            padding2 = resize_shape[0] - input_shape[0]
            assert padding2 % 2 == 0
            padding = padding2 // 2
        else:
            padding = 0

        if resize_shape and resize_shape[2] != input_shape[2]:
            self.reshape_mlp = nn.Conv2d(input_shape[2], resize_shape[2], kernel_size=1)

        in_shape = resize_shape if resize_shape else input_shape
        self.proj = nn.Conv2d(in_shape[2], embed_dim, kernel_size=in_shape[0], padding=padding)
        if pretrained:
            self.init_weights(pretrained)

    def init_weights(self, pretrained):
        # only load self.mlp
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        old_dict = torch.load(pretrained, map_location='cpu')['state_dict']
        mlp_weight_key = 'patch_embed.proj'
        weight_names = ('weight', 'bias')
        state_dict = dict()
        for name in weight_names:
            state_dict[f'proj.{name}'] = old_dict[f'{mlp_weight_key}.{name}']
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, inputs):
        if isinstance(inputs, list):
            outs = inputs # fuck this, but I'm lazy
            # resize
            if hasattr(self, 'reshape_mlp'):
                outs = [self.reshape_mlp(x) for x in outs]

            if self.resize_shape and self.resize_method == 'bilinear':
                outs = [nn.functional.interpolate(x, size=self.resize_shape[:2], mode='bilinear') for x in outs]

            outs = [self.proj(x) for x in outs]   # [ (B, 768, 1, 1)]
            outs = torch.cat(outs, dim=2)   # (B, C, n, 1)
            outs = outs.squeeze(3).transpose(2, 1)
        else:
            raise TypeError('ConvFlatten inputs should be list')
        return outs


@NECKS.register_module()
class GlobalAvgPoolingFlatten(nn.Module):
    """
    use Conv2D and AvgPooling to flatten (C,H,W)
    input:
        input_shape: h, w, c
        x: [ Tensor((B, 1, H, W)), ]
    
    process:
        Conv2d(kernel=3, stride=2)
        AveragePooling

    output: Tensor((B, n, c)), n = len(x)
    """
    def __init__(self, input_shape, embed_dim):
        super().__init__()
        assert len(input_shape) == 3, "input_shape must have 3 dimensions"
        self.conv = nn.Conv2d(input_shape[2], embed_dim, kernel_size=3, stride=2)

    def forward(self, x):
        if isinstance(x, list):
            x = [self.conv(i) for i in x]
            x = [nn.AdaptiveAvgPool2d(1)(i) for i in x]# [ (B, 768, 1, 1)]

            outs = torch.cat(x, dim=2)   # (B, C, n, 1)
            outs = outs.squeeze(3).transpose(2, 1)
        else:
            raise TypeError('GlobalAvgPoolingFlatten inputs should be list')
        return outs


@NECKS.register_module()
class MaxHybridFlatten(nn.Module):
    """
    input:
        input_shape: h, w, c
        x: [ Tensor((B, 1, H, W)), ]
    
    process:
        max(multi LA feature)
        1x1 Conv

    output: Tensor((B, n, c)), n = len(x)
    """
    def __init__(self, input_shape, embed_dim, drop_rate=0, drop_upper=0, adaptive_drop=False, pretrained=None,
        pool_config=dict(keep_num=10, alpha1=1, alpha2=0, exclude_first=False),
        ):
        super().__init__()
        assert len(input_shape) == 3, "input_shape must have 3 dimensions"
        assert 0 <= drop_rate <= 1 
        self.proj = nn.Conv2d(input_shape[2], embed_dim, kernel_size=1)
        self.drop_rate = drop_rate
        self.drop_upper = drop_upper
        self.adaptive_drop = adaptive_drop
        if pretrained:
            self.init_weights(pretrained)
        if adaptive_drop:
            print('MaxHybridFlatten Adaptive Drop enabled')
        self.pool_config = pool_config

    def init_weights(self, pretrained):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        old_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in old_dict:
            old_dict = old_dict['state_dict']
        mlp_weight_key = 'patch_embed.proj'
        weight_names = ('weight', 'bias')
        state_dict = dict()
        for name in weight_names:
            state_dict[f'proj.{name}'] = old_dict[f'{mlp_weight_key}.{name}']
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, feature, la_outs, **kwargs):
        keep_index = None
        if la_outs is None:
            x = self.proj(feature)    # [ (B, 768, H, W)]
            outs = torch.flatten(x, start_dim=2)   # (B, 768, HxW)
            outs = outs.transpose(2, 1)
        elif isinstance(la_outs, list):
            x = torch.cat(la_outs, dim=1)  #(B N H W)
            B, N, _, _ = x.shape
            # dropout
            if self.training and self.drop_rate > 0 and self.drop_upper > 0:
                batch_rand = torch.rand((B, ))
                batch_mask = batch_rand <= self.drop_rate
                all_indexs = [i for i in range(N)]
                if self.adaptive_drop:
                    weights = nn.AdaptiveAvgPool2d(1)(x).reshape(B, N)
                    weights = torch.add(torch.sqrt(weights), 1e-12)
                    weights = torch.div(weights, torch.sum(weights, dim=1).unsqueeze(1))
                    # weights += weights.min()
                else:
                    weights = [None] * B

                for i in range(B):
                    if batch_mask[i]:
                        k = random.randint(1, self.drop_upper)
                        zero_index = random.choices(all_indexs, k=k, weights=weights[i])
                        x[i, zero_index, ...] = 0

            x, _ = torch.max(x, dim=1, keepdim=True) #(B 1 H W)
            feature = self.proj(feature)    # [ (B, 768, H, W)]
            feature = feature * x
            outs = torch.flatten(feature, start_dim=2)   # (B, 768, HxW)
            if self.pool_config:
                atten = torch.flatten(x, start_dim=1).unsqueeze(dim=-1)   # (B, HxW, 1)
                keep_index = top_pool(atten, **self.pool_config)

            outs = outs.transpose(1, 2)
        else:
            raise TypeError('MaxHybridFlatten inputs should be list')
        return dict(x=outs, keep_index=keep_index)


@NECKS.register_module()
class ImgCropFlatten(nn.Module):
    """
    crop image according to LA output, and then resize to same size.
    input:
        input_shape: h, w, c
        x: [ Tensor((B, 1, H, W)), ]
    
    process:
        max(multi LA feature)
        1x1 Conv

    output: Tensor((B, n, c)), n = len(x)
    """
    def __init__(self, input_shape, embed_dim, lower, upper, sum_thr=5, pretrained=None):
        super().__init__()
        assert len(input_shape) == 3, "input_shape must have 3 dimensions"
        self.lower = lower
        self.upper = upper
        self.sum_thr = sum_thr
        self.proj = nn.Conv2d(input_shape[2], embed_dim, kernel_size=16)
        if pretrained:
            self.init_weights(pretrained)

    def init_weights(self, pretrained):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        old_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in old_dict:
            old_dict = old_dict['state_dict']
        mlp_weight_key = 'patch_embed.proj'
        weight_names = ('weight', 'bias')
        state_dict = dict()
        for name in weight_names:
            state_dict[f'proj.{name}'] = old_dict[f'{mlp_weight_key}.{name}']
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def get_bbox_by_mask(self, mask):
        """
        return: top, left, height, width
        """
        assert mask.dim() == 2, f'mask shape should be (H, W), but get {mask.shape}'
        def get_change_point(row):
            left, right = 0, len(row)-1
            for i in range(len(row)):
                if row[i] == True:
                    left = i
                    break
            for i in range(len(row)-1, -1, -1):
                if row[i] == True:
                    right = i
                    break
            if right - left <= 3:
                print(left, right)
                if left <= 3:
                    right += 3
                elif right >= len(row) - 4:
                    left -= 3
                else:
                    left = max(0, left-2)
                    right = min(len(row)-1, right+2)
                # return None
            return left, right
        width_ = get_change_point(mask.sum(0) >= self.sum_thr)
        height_ = get_change_point(mask.sum(1) >= self.sum_thr)
        if width_ and height_:
            return height_[0], width_[0], height_[1]-height_[0], width_[1]-width_[0]
        return

    def forward(self, img, la_outs, **kwargs):
        if isinstance(la_outs, list):
            B = img.shape[0]
            # cast la to bool
            threshold = random.uniform(self.lower, self.upper)
            masks = [i >= threshold for i in la_outs]   # [(B, 1, H, W)]
            out_batch = list()
            for i in range(B):
                masks_in = [m[i, 0, ...] for m in masks]
                # im = ToPILImage()(img[i].cpu())
                im = img[i]
                # get bbox
                bboxs = [self.get_bbox_by_mask(i) for i in masks_in]
                bboxs = [i for i in bboxs if i]
                # resize
                # imgs = [torchvision.transforms.functional.crop(im, *bbox) for bbox in bboxs]
                # imgs = [torchvision.transforms.functional.resize(im, (16, 16)) for im in imgs]    # [B C H W]
                imgs = [im[:, bbox[0]: bbox[0]+bbox[2], bbox[1]: bbox[1]+bbox[3]] for bbox in bboxs]
                imgs = [im.unsqueeze(0) for im in imgs]
                imgs = [torch.nn.functional.interpolate(im, (16, 16), mode='bilinear') for im in imgs]
                # proj
                # imgs = [ToTensor()(im).unsqueeze(0).cuda() for im in imgs]
                x = [self.proj(i) for i in imgs]    # [B 768 1 1]
                outs = torch.cat(x, dim=2)   # (B, C, n, 1)
                outs = outs.squeeze(3).transpose(2, 1)
                out_batch.append(outs)
            outs = torch.cat(out_batch, dim=0)
        else:
            raise TypeError('inputs should be list')
        return outs



@NECKS.register_module()
class LANetAttention(nn.Module):
    """
    Pool the feature map with attention generated by ONE LANet.

    process:
        for every input:
            1. project the feature map to a certain dimension.
            2. generate an attention map by ONE LANet
            3. pooling the feature map by the attention map

    """
    def __init__(self, in_channels:List[int], out_channel:int, ratios:List[int], patch_nums:List[int],
        pool_configs=None,
        pretrained=None):
        super().__init__()
        assert len(in_channels) == len(ratios) == len(pool_configs), 'configs must have the same length'
        self.projs = nn.ModuleList([nn.Conv2d(in_channel, out_channel, kernel_size=1) for in_channel in in_channels])
        self.las =  nn.ModuleList([LANet(in_channel, ratio) for in_channel, ratio in  zip(in_channels, ratios)])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, patch_num, out_channel), requires_grad=True) for patch_num in patch_nums])
        # if len(pool_configs) >= 2: # TODO, only add stage pos embed when len >= 2
        if len(in_channels) == 2:
            if pool_configs[0]['keep_num'] > 0:
                self.stage_pos_embeds = nn.Parameter(torch.tensor([0., 1.]), requires_grad=True)
        elif len(in_channels) == 3:
            self.stage_pos_embeds = nn.Parameter(torch.tensor([0., 0., 1.]), requires_grad=True)

        self.pool_configs = pool_configs
        if pretrained is not None:
            self.init_weights(pretrained)
    
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed'][:, 1:, ...]     # [1, 197, 768] for base
        pos_embed_params = []
        for pos_embed_new in self.pos_embeds:
            patch_num_new = pos_embed_new.shape[1]
            if patch_num_new != pos_embed.shape[1]:
                logger.warning(f'interpolate pos_embed from {pos_embed.shape[1]} to {patch_num_new}')
                pos_embed_new = resize_pos_embed(pos_embed, pos_embed_new, 0)
                pos_embed_params.append(pos_embed_new)  # skip cls_token
            else:
                pos_embed_params.append(pos_embed)
        load_state_dict(self, dict(pos_embeds=pos_embed_params), strict=False, logger=logger)
    
    def forward(self, x:List[Tensor]):
        assert len(x) == len(self.projs)
        tokens = [self.projs[i](x[i]) for i in range(len(x))]
        las_2D = [self.las[i](x[i]) for i in range(len(x))]
        tokens = [i.flatten(start_dim=2).transpose(1, 2) for i in tokens]
        las = [i.flatten(start_dim=2).transpose(1, 2) for i in las_2D]
        tokens = [tokens[i] * las[i] + self.pos_embeds[i] for i in range(len(tokens))]
        # apply stage position embedding
        if hasattr(self, 'stage_pos_embeds'):
            tokens = [tokens[i] * self.stage_pos_embeds[i] for i in range(len(tokens))]
            if not self.training:
                print(self.stage_pos_embeds[0].item(), self.stage_pos_embeds[1].item())

        keep_indexes = [
            top_pool(las[i], **self.pool_configs[i])
            for i in range(len(tokens))
        ]
        results = []
        for i in range(len(tokens)):
            if keep_indexes[i] is None:
                results.append(tokens[i])
                continue
            stage_out = []
            B, N, C = tokens[i].shape
            for j in range(B):
                stage_out.append(tokens[i][j, keep_indexes[i][j], ...])
            results.append(torch.stack(stage_out))
        # results = torch.cat(results, dim=1)
        return dict(x=results, la_outs=las_2D)



@NECKS.register_module()
class LANetAttentionMaxPooling(nn.Module):
    """
    Pool the feature map with attention generated by ONE LANet on S3.
    The input shoule be the second (S2) and the third stage (S3) of the feature map

    process:
        generate an attention map by ONE LANet
        MaxPooling the S2 to have the same size as S3
        for every input:
            1. project the feature map to a certain dimension.
            3. pooling the feature map by the attention map if the score is greater than pool_threshold

    """
    def __init__(self, in_channels:List[int], out_channel:int, patch_num, la_ratio=16,
        pool_config=None,
        pretrained=None):
        super().__init__()
        self.projs = nn.ModuleList([nn.Conv2d(in_channel, out_channel, kernel_size=1) for in_channel in in_channels])
        self.la = LANet(in_channels[-1], la_ratio)
        self.pos_embeds = nn.Parameter(torch.zeros(1, patch_num, out_channel), requires_grad=True)
        self.s1_pooling = nn.MaxPool2d(kernel_size=4)
        self.s2_pooling = nn.MaxPool2d(kernel_size=2)

        self.pool_config = pool_config
        if pretrained is not None:
            self.init_weights(pretrained)
    
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        logger.warning(f'{self.__class__.__name__} load pretrain from {pretrained}')
        state_dict = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        pos_embed = state_dict['pos_embed'][:, 1:, ...]     # [1, 197, 768] for small

        patch_num_new = self.pos_embeds.shape[1]
        if patch_num_new != pos_embed.shape[1]:
            logger.warning(f'interpolate pos_embed from {pos_embed.shape[1]} to {patch_num_new}')
            pos_embed_new = resize_pos_embed(pos_embed, self.pos_embeds.shape, 0)
        else:
            pos_embed_new = pos_embed
        load_state_dict(self, dict(pos_embeds=pos_embed_new), strict=False, logger=logger)
    
    def forward(self, x:List[Tensor]):
        assert len(x) == len(self.projs), f'{len(x)}, {len(self.projs)}'
        B, C, H, W = x[-1].shape
        if len(x) == 1:
            pass
        elif len(x) == 2:
            x[0] = self.s2_pooling(x[0]).reshape(B, -1, H, W)
        elif len(x) == 3:
            x[0] = self.s1_pooling(x[0]).reshape(B, -1, H, W)
            x[1] = self.s2_pooling(x[0]).reshape(B, -1, H, W)

        la_2D = self.la(x[-1])
        tokens = [self.projs[i](x[i]) for i in range(len(x))]   # B, 1, H, W

        tokens = [i.flatten(start_dim=2).transpose(1, 2) for i in tokens]
        la = la_2D.flatten(start_dim=2).transpose(1, 2)
        tokens = [tokens[i] * la + self.pos_embeds/len(x) for i in range(len(tokens))]

        if self.pool_config is not None:
            keep_indexes = top_pool(la, **self.pool_config)

            results = []
            for i in range(len(tokens)):
                stage_out = []
                B, N, C = tokens[i].shape
                for j in range(B):
                    if keep_indexes is not None:
                        stage_out.append(tokens[i][j, keep_indexes[j], ...])
                    else:
                        stage_out.append(tokens[i][j, ...])
                results.append(torch.stack(stage_out))

            return dict(x=results, la_outs=[la_2D])
        else:
            return dict(x=tokens, la_outs=[la_2D])




@NECKS.register_module()
class LANetAttentionUnfold(LANetAttentionMaxPooling):
    """
    Pool the feature map with attention generated by ONE LANet on S3.
    The input shoule be the second (S2) and the third stage (S3) of the feature map

    process:
        generate an attention map by ONE LANet
        Unfold the S2 to have the same size as S3, and quadruple feature dims
        for every input:
            1. project the feature map to a certain dimension.
            3. pooling the feature map by the attention map if the score is greater than pool_threshold

    """
    def __init__(self, in_channels:List[int], out_channel:int, patch_num, la_ratio=16,
        pool_config=None,
        pretrained=None):
        super().__init__(in_channels, out_channel, patch_num, la_ratio, pool_config, pretrained)
        self.s1_pooling = nn.Unfold(kernel_size=4, stride=4)
        self.s2_pooling = nn.Unfold(kernel_size=2, stride=2)    # Unfold instead of MaxPooling




