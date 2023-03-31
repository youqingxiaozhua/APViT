# dataset settings
dataset_type = 'RAF'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_size = 112

train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', size=img_size),
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
    dict(type='RandomErasing', p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=img_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=['gt_label', ]),
    dict(type='Collect', keys=['img', ])
]

base_path = 'data/RAF-DB/basic/'
image_path = base_path + 'Image/aligned_224'    # we use realigned images

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.15,
        method='sqrt',
        dataset=dict(
            type=dataset_type,
            data_prefix=image_path,
            ann_file=base_path + 'EmoLabel/train.txt',
            pipeline=train_pipeline),
    ),
    val=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + 'EmoLabel/test.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_prefix=image_path,
        ann_file=base_path + 'EmoLabel/test.txt',
        pipeline=test_pipeline),
)

lr_config = dict()

workflow = [('train', 1), ]
evaluation = dict(interval=1, metric=['accuracy', 'class_accuracy'])
checkpoint_config = dict(create_symlink=False, max_keep_ckpts=1, interval=100)
runner = dict(type='EpochBasedRunner', max_epochs=40)
