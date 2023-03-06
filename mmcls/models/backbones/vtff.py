import torch
import torch.nn as nn

from ..builder import BACKBONES, build_backbone
from .base_backbone import BaseBackbone


"""
Coarse implementation of VTFFF to count FLOPs.

"""


class AFS(nn.Module):
    def __init__(self,):
        super().__init__()
        self.Wl = nn.Conv2d(512, 512, 1, 1)
        self.Wc = nn.Conv2d(512, 512, 1, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.convG1 = nn.Conv2d(512, 64, 1, 1)  # 512 // 8
        self.convG2 = nn.Conv2d(64, 512, 1, 1)
        self.bnG1 = nn.BatchNorm2d(64)
        self.bnG2 = nn.BatchNorm2d(512)

        self.convL1 = nn.Conv2d(512, 64, 1, 1)
        self.convL2 = nn.Conv2d(64, 1, 1, 1)
        self.bnL1 = nn.BatchNorm2d(64)
        self.bnL2 = nn.BatchNorm2d(1)

    def forward(self, x1, x2):
        U = self.Wl(x1) + self.Wc(x2)

        GU = self.sigmoid(self.bnG2(self.convG2(self.sigmoid(self.bnG1(self.convG1(nn.AdaptiveAvgPool2d((1, 1))(U)))))))
        LU = self.sigmoid(self.bnL2(self.convL2(self.sigmoid(self.bnL1(self.convL1(U))))))
        GL = GU + LU
        x = x2 * self.sigmoid(GL) + x1 * self.sigmoid(1 - GL)
        return x



@BACKBONES.register_module()
class VTFF(BaseBackbone):
    """
    paper: Facial Expression Recognition with Visual Transformers and Attentional Selective Fusion
    """

    def __init__(self, ):
        super().__init__()
        cnn1 = dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch')
        cnn2 = dict(
            type='ResNet',
            depth=18,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch')
        vit = dict(
            type='VisionTransformerOrigin',
            input_type='feature',
            num_patches=49,
            depth=8,
            last_ln=True,
            img_size=112, patch_size=16, embed_dim=768, num_heads=8, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6
        )

        self.cnn1 = build_backbone(cnn1)
        self.cnn2 = build_backbone(cnn2)
        self.afs = AFS()
        self.project = nn.Conv2d(512, 768, 1, 1)
        self.vit = build_backbone(vit)

    def forward(self, x):

        x1 = self.cnn1(x)[0]
        x2 = self.cnn2(x)[0]

        x_fused = self.afs(x1, x2)

        x = self.project(x_fused)
        x = x.flatten(2)
        x = x.transpose(1, 2)  # [B, N, C]
        x = self.vit(x)
        
        return x


@BACKBONES.register_module()
class MViT(BaseBackbone):
    """
    paper: MViT: Mask Vision Transformer for Facial Expression Recognition in the wild
    """

    def __init__(self, ):
        super().__init__()
        generator = dict(   # DeiT-S first 6 layers
            type='VisionTransformerOrigin',
            input_type='image',
            # num_patches=49,
            depth=6,
            last_ln=True,
            img_size=224, patch_size=16, embed_dim=384, num_heads=6, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6
        )
        mvit = dict(   # DeiT-S
            type='VisionTransformerOrigin',
            input_type='image',
            # num_patches=14 ** 2,
            depth=12,
            last_ln=True,
            img_size=224, patch_size=16, embed_dim=384, num_heads=6, mlp_ratio=3, qkv_bias=False, norm_layer_eps=1e-6
        )
        self.generator = build_backbone(generator)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(384, 768)
        self.mvit = build_backbone(mvit)
    
    def forward(self, img):
        x = self.generator(img)
        B, N, C = x.shape
        output = []
        for i in range(1, N):
            output.append(self.sigmoid(self.proj(x[:, i, :])))
        x = torch.stack(output) # [B, C, N]
        x = x.reshape((B, 3, 224, 224))

        x = x * img

        x = self.mvit(x)
        return x[:, 0]


