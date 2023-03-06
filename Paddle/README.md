
## TransFER

This is the PaddlePaddle implement of our ICCV paper [TransFER: Learning Relation-aware Facial Expression Representations with Transformers](https://openaccess.thecvf.com/content/ICCV2021/html/Xue_TransFER_Learning_Relation-Aware_Facial_Expression_Representations_With_Transformers_ICCV_2021_paper.html)


### Test

1. First, download the pre-trained weight from [Baidu Netdisk](https://pan.baidu.com/s/1nuSigUoyV2qEB6WMtzUWFQ?pwd=axfc)
2. Put the RAF-DB dataset on `data/RAF-DB`
3. Run:
```
python tools/eval.py \
    -c ./ppcls/configs/RAF/TransFER.yaml \
    -o Global.pretrained_model=TransFER-RAF_param
```

You will see the output like this:
```
W1030 17:05:54.343888 36757 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.2, Runtime API Version: 10.2
W1030 17:05:54.393682 36757 device_context.cc:422] device: 0, cuDNN Version: 7.6.
load from output/RAFa_2_6131_ep33_90p91_param
[2021/10/30 17:06:04] root INFO: train with paddle 2.1.2 and device CUDAPlace(0)
{'CELoss': {'weight': 1.0}}
[2021/10/30 17:06:05] root INFO: [Eval][Epoch 0][Iter: 0/48]CELoss: 0.46290, loss: 0.46290, top1: 0.89062, top3: 0.98438, batch_cost: 1.17835s, reader_cost: 0.85988, ips: 54.31341 images/sec
[2021/10/30 17:06:08] root INFO: [Eval][Epoch 0][Iter: 10/48]CELoss: 0.17079, loss: 0.17079, top1: 0.95312, top3: 0.98438, batch_cost: 0.31895s, reader_cost: 0.00042, ips: 200.65537 images/sec
[2021/10/30 17:06:12] root INFO: [Eval][Epoch 0][Iter: 20/48]CELoss: 0.37811, loss: 0.37811, top1: 0.90625, top3: 0.98438, batch_cost: 0.31878s, reader_cost: 0.00098, ips: 200.76747 images/sec
[2021/10/30 17:06:15] root INFO: [Eval][Epoch 0][Iter: 30/48]CELoss: 0.28195, loss: 0.28195, top1: 0.92188, top3: 1.00000, batch_cost: 0.32863s, reader_cost: 0.00104, ips: 194.74883 images/sec
[2021/10/30 17:06:18] root INFO: [Eval][Epoch 0][Iter: 40/48]CELoss: 0.35353, loss: 0.35353, top1: 0.90625, top3: 1.00000, batch_cost: 0.32664s, reader_cost: 0.00101, ips: 195.93285 images/sec
[2021/10/30 17:06:20] root INFO: [Eval][Epoch 0][Avg]CELoss: 0.33157, loss: 0.33157, top1: 0.90906, top3: 0.98827
```



