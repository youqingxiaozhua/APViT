APViT: Vision Transformer With Attentive Pooling for Robust Facial Expression Recognition
==

<div align="center">
  <img src="resources/model.png" width="800"/>
</div>

APViT is a simple and efficient Transformer-based method for facial expression recognition (FER). It builds on the [TransFER](https://openaccess.thecvf.com/content/ICCV2021/html/Xue_TransFER_Learning_Relation-Aware_Facial_Expression_Representations_With_Transformers_ICCV_2021_paper.html), but introduces two attentive pooling (AP) modules that do not require any learnable parameters. These modules help the model focus on the most expressive features and ignore the less relevant ones. You can read more about our method in our [paper](https://arxiv.org/abs/2212.05463).


## Installation

This project is based on [MMClassification](https://github.com/open-mmlab/mmclassification) and [PaddleClas](https://github.com/PaddlePaddle/PaddleClas), please refer to their repos for installation and dataset preparation.

Notable, our method does not rely on custome cuda operations in mmcv-full.

The pre-trained weight of IR-50 weight was downloaded from [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe/#model-zoo), and 
ViT-Small was downloaded from [pytorch-image-models](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth). 

### PaddlePaddle Version

The PaddlePaddle version of TransFER is included in the [paddle](paddle) folder.


## Training

To train an APViT model with two GPUs, use:

```shell
python -m torch.distributed.launch --nproc_per_node=2 \
    train.py configs/apvit/RAF.py \
    --launcher pytorch
```

## Evaluation

To evaluate the model with a given checkpoint, use:

```shell
PYTHONPATH=$(pwd):$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=2 \
    tools/test.py configs/apvit/RAF.py \
    weights/APViT_RAF-3eeecf7d.pth \   # your checkpoint
    --launcher pytorch
```

## Pretrained checkpoints

| Model | RAF-DB | Config   | Download |
|-------|--------|----------|----------|
| APViT | 91.98%  | [config](configs/apvit/RAF.py) | [model](https://pan.baidu.com/s/1nuSigUoyV2qEB6WMtzUWFQ?pwd=axfc)  |


## License

This project is released under the [Apache 2.0 license](LICENSE).


## Reference
If you use APViT or TransFER, please cite the paper:

```
@article{xue2022vision,
  title={Vision Transformer with Attentive Pooling for Robust Facial Expression Recognition},
  author={Xue, Fanglei and Wang, Qiangchang and Tan, Zichang and Ma, Zhongsong and Guo, Guodong},
  journal={IEEE Transactions on Affective Computing},
  year={2022},
  publisher={IEEE}
}

@inproceedings{xue2021transfer,
  title={Transfer: Learning Relation-aware Facial Expression Representations with Transformers},
  author={Xue, Fanglei and Wang, Qiangchang and Guo, Guodong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3601--3610},
  year={2021}
}
```
