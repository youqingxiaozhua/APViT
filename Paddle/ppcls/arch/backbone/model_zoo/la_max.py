import logging
import random
import paddle
from paddle import nn


class LANet(nn.Layer):

    def __init__(self, channel_num, ratio=16):
        super().__init__()
        assert channel_num % ratio == 0, f'input_channel{channel_num} must be exact division by ratio{ratio}'
        self.channel_num = channel_num
        self.ratio = ratio
        self.relu = paddle.nn.ReLU()
        self.LA_conv1 = nn.Conv2D(channel_num, int(channel_num / ratio),
            kernel_size=1)
        self.bn1 = nn.BatchNorm2D(int(channel_num / ratio))
        self.LA_conv2 = nn.Conv2D(int(channel_num / ratio), 1, kernel_size=1)
        self.bn2 = nn.BatchNorm2D(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        LA = self.LA_conv1(x)
        LA = self.bn1(LA)
        LA = self.relu(LA)
        LA = self.LA_conv2(LA)
        LA = self.bn2(LA)
        LA = self.sigmoid(LA)
        return LA


class LANetAttention_div_loss(paddle.nn.Layer):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, x):
        """
        x: numbers of attention maps, shape(H, W)
        [Tensor((B, C, H, W)), ]
        """
        loss = 0.0
        num = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                x0 = paddle.squeeze(x[i])
                x1 = paddle.squeeze(x[j])
                diff = x0 - x1
                dist_sq = paddle.sum(paddle.sum(diff ** 2, 1), 1)
                dist = paddle.sqrt(dist_sq) / (x0.size()[2] * x0.size()[1])
                mdist = self.margin - dist
                dist = paddle.clip(mdist, min=0.0)
                loss += paddle.sum(dist) / 2.0 / x0.size()[0]
                num += 1
        return loss / num


class MultiLANet(nn.Layer):
    """Multi LANet.
    from paper: Hierarchical pyramid diverse attention networks for face recognition
    
    return: [Tensor(B,1,H,W), ], loss
    """

    def __init__(self, in_channels, ratio=16, num=5, loss_margin=1.0,
        loss_weight=0.):
        super().__init__()
        self.num = num
        self.loss_weight = loss_weight
        self.criterion = LANetAttention_div_loss(loss_margin)
        self.las = nn.LayerList([LANet(in_channels, ratio) for _ in range(num)]
            )

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            raise TypeError('tuple inputs are not suported by MultiLANet now')
        elif isinstance(inputs, paddle.Tensor):
            if self.num == 0:
                return dict(feature=inputs, la_outs=None, loss=paddle.
                    to_tensor(data=[0.0], place='cuda'))
            la_outs = [f(inputs) for f in self.las]
            if self.loss_weight > 0:
                loss = self.criterion(la_outs) * self.loss_weight
            else:
                loss = paddle.to_tensor(data=[0.0])
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return dict(feature=inputs, la_outs=la_outs, loss=loss)


class MaxHybridFlatten(nn.Layer):
    """
    input:
        input_shape: h, w, c
        x: [ Tensor((B, 1, H, W)), ]
    
    process:
        max(multi LA feature)
        1x1 Conv

    output: Tensor((B, n, c)), n = len(x)
    """

    def __init__(self, input_shape, embed_dim, drop_rate=0, drop_upper=0,
        adaptive_drop=False, pretrained=None):
        super().__init__()
        assert len(input_shape) == 3, 'input_shape must have 3 dimensions'
        assert 0 <= drop_rate <= 1
        self.proj = nn.Conv2D(input_shape[2], embed_dim, kernel_size=1)
        self.drop_rate = drop_rate
        self.drop_upper = drop_upper
        self.adaptive_drop = adaptive_drop
        if pretrained:
            self.init_weights(pretrained)
        if adaptive_drop:
            print('MaxHybridFlatten Adaptive Drop enabled')

    def init_weights(self, pretrained):
        logger = logging.getLogger()
        logger.warning(
            f'{self.__class__.__name__} load pretrain from {pretrained}')
        old_dict = paddle.load(pretrained)
        if 'state_dict' in old_dict:
            old_dict = old_dict['state_dict']
        mlp_weight_key = 'patch_embed.proj'
        weight_names = 'weight', 'bias'
        state_dict = dict()
        for name in weight_names:
            state_dict[f'proj.{name}'] = old_dict[f'{mlp_weight_key}.{name}']
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def forward(self, feature, la_outs, **kwargs):
        if la_outs is None:
            x = self.proj(feature)
            outs = x.flatten(start_axis=2)
            outs = outs.transpose(2, 1)
        elif isinstance(la_outs, list):
            x = paddle.concat(la_outs, axis=1)
            B, N, _, _ = x.shape
            if self.training and self.drop_rate > 0 and self.drop_upper > 0:
                batch_rand = paddle.rand((B,))
                batch_mask = (batch_rand <= self.drop_rate).numpy()
                all_indexs = [i for i in range(N)]
                if self.adaptive_drop:
                    weights = nn.AdaptiveAvgPool2D(1)(x).reshape(B, N)
                    weights = paddle.add(paddle.sqrt(weights), 1e-12)
                    weights = (weights / paddle.sum(weights, axis=1)
                        ).unsqueeze(1)
                else:
                    weights = [None] * B
                for i in range(B):
                    if batch_mask[i]:
                        k = random.randint(1, self.drop_upper)
                        zero_index = random.choices(all_indexs, k=k,
                            weights=weights[i])
                        x[i, zero_index, ...] = 0
            x = paddle.max(x, axis=1, keepdim=True)
            feature = feature * x
            x = self.proj(feature)
            outs = x.flatten(start_axis=2)
            # outs = outs.transpose(1, 2)
            outs = paddle.transpose(outs, perm=(0, 2, 1))
        else:
            raise TypeError('MaxHybridFlatten inputs should be list')
        return outs
