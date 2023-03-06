# dataset settings
dataset_type = 'CIFAR10'
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=True)
img_size = 112
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=224),
    dict(type='RandomRotate', prob=0.5, degree=6),
    dict(type='RandomResizedCrop', size=img_size, scale=(0.8, 1.0), ratio=(1. / 1., 1. / 1.)),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='RandomGrayscale', gray_prob=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomErasing', p=0.5,  scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=(256, -1)),
    dict(type='Resize', size=img_size),
    # dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label', ]),
    dict(type='Collect', keys=['img', 'gt_label',])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type, data_prefix='data/cifar',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type, data_prefix='data/cifar', pipeline=test_pipeline),
    test=dict(
        type=dataset_type, data_prefix='data/cifar', pipeline=test_pipeline))

workflow = [('train', 1), ]
evaluation = dict(interval=1, metric=['accuracy',])
checkpoint_config = dict(create_symlink=False, max_keep_ckpts=1, interval=100)
runner = dict(type='EpochBasedRunner', max_epochs=40)

