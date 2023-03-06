# checkpoint saving
# checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='VisualDLLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 1), ]

# use_fp16 = True
# optimizer_config = dict(use_fp16=use_fp16)
# fp16 = dict(loss_scale='dynamic')
