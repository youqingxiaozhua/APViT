import torch
import torch.nn as nn

from ..builder import CLASSIFIERS, build_backbone, build_head, build_neck
from .base import BaseClassifier



def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention




@CLASSIFIERS.register_module()
class AttentionVitClassifier(BaseClassifier):

    def __init__(self, extractor, attention, convert, vit, neck=None, head=None, pretrained=None, freeze_backbone=False):
        super().__init__()
        if extractor:
            self.extractor = build_backbone(extractor)
        if freeze_backbone and extractor:
            print('freeze backbone: %s' % extractor['type'])
            self.extractor.eval()
            for param in self.extractor.parameters():
                    param.requires_grad = False
        self.attention = build_neck(attention)
        self.convert = build_neck(convert)
        self.vit:nn.Module = build_backbone(vit)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        # self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_head:
            self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone + neck
        """
        aux_loss = dict()
        if hasattr(self, 'extractor'):
            x = self.extractor(img)
        else:
            x = img
        # print(x.shape)
        # exit()
        x = self.attention(x)
        aux_loss.update(x['loss'])
        x = self.convert(img=img, **x)
        x = self.vit(**x)
        aux_loss.update(x['loss'])
        x = x['x']
        if self.with_neck:
            x = self.neck(x)
        return x, aux_loss
    
    def extract_la(self, img):
        if hasattr(self, 'extractor'):
            x = self.extractor(img)
        else:
            x = img
        x = self.attention(x)
        return x['la_outs']

    def forward_train(self, img, gt_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            gt_label (Tensor): of shape (N, 1) encoding the ground-truth label
                of input images.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x, aux_loss = self.extract_feat(img)

 
        losses = self.head.forward_train(x, gt_label)
        # losses['ce_loss'] = losses['loss']
        # losses['loss'] *= 0.
        # losses['aux_loss'] = aux_loss
        losses.update(aux_loss)

        return losses

    def simple_test(self, img, **kwargs):
        """Test without augmentation."""
        x, _ = self.extract_feat(img)
        return self.head.simple_test(x)
    
    def inference(self, img, **kwargs):
        x, _ = self.extract_feat(img)
        x = self.head.extract_feat(x)
        return x
    
    def aug_test(self, imgs, **kwargs): # TODO: pull request: add aug test to mmcls
        logit = self.inference(imgs[0], **kwargs)
        for i in range(1, len(imgs)):
            cur_logit = self.inference(imgs[i])
            logit += cur_logit
        logit /= len(imgs)
        # pred = F.softmax(logit, dim=1)
        pred = logit
        pred = pred.cpu().numpy()
        # unravel batch dim
        pred = list(pred)
        return pred

    def relprop(self, cam=None,method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        cam = self.vit.pool.relprop(cam, **kwargs)
        cam = self.vit.norm.relprop(cam, **kwargs)
        for blk in reversed(self.vit.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.vit.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.vit.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.vit.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam
        
        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.vit.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            cam = rollout[:, 0, 1:]
            # cam = rollout[:, 1:, 0]
            return cam
            
        elif method == "last_layer":
            cam = self.vit.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.vit.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.vit.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.vit.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.vit.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam


