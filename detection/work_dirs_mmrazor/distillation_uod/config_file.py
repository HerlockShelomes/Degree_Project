auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
data_root = '/data/fxy_datasets/underwater_bs_deblur/'
dataset_type = 'BSDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet', interval=1, max_keep_ckpts=2, type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
find_unused_parameters = True
launcher = 'pytorch'
load_from = '/data/fxy_projects/mmdetection-main/work_dirs/fcos_bs_deblur/best_mAP_epoch_12_teacher.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    _scope_='mmrazor',
    architecture=dict(
        cfg_path=
        'mmdet::fcos/fcos_r18_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py',
        pretrained=False),
    distiller=dict(
        distill_losses=dict(
            loss_cwd_fpn0=dict(
                loss_weight=10, tau=1, type='ChannelWiseDivergence'),
            loss_cwd_fpn1=dict(
                loss_weight=10, tau=1, type='ChannelWiseDivergence'),
            loss_cwd_fpn2=dict(
                loss_weight=10, tau=1, type='ChannelWiseDivergence'),
            loss_cwd_fpn3=dict(
                loss_weight=10, tau=1, type='ChannelWiseDivergence'),
            loss_cwd_fpn4=dict(
                loss_weight=10, tau=1, type='ChannelWiseDivergence')),
        loss_forward_mappings=dict(
            loss_cwd_fpn0=dict(
                preds_S=dict(data_idx=0, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=0, from_student=False, recorder='fpn')),
            loss_cwd_fpn1=dict(
                preds_S=dict(data_idx=1, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=1, from_student=False, recorder='fpn')),
            loss_cwd_fpn2=dict(
                preds_S=dict(data_idx=2, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=2, from_student=False, recorder='fpn')),
            loss_cwd_fpn3=dict(
                preds_S=dict(data_idx=3, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=3, from_student=False, recorder='fpn')),
            loss_cwd_fpn4=dict(
                preds_S=dict(data_idx=4, from_student=True, recorder='fpn'),
                preds_T=dict(data_idx=4, from_student=False, recorder='fpn'))),
        student_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        teacher_recorders=dict(fpn=dict(source='neck', type='ModuleOutputs')),
        type='ConfigurableDistiller'),
    teacher=dict(cfg_path='mmdet::fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py'),
    type='FpnTeacherDistill')
optim_wrapper = dict(
    _scope_='mmdet',
    optimizer=dict(lr=0.02, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        _scope_='mmdet',
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
teacher_ckpt = '/data/fxy_projects/mmdetection-main/work_dirs/fcos_bs_deblur/best_coco_bbox_mAP_epoch_12.pth'
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='test_annotations_clean.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/data/fxy_datasets/underwater_bs_deblur/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='BSDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file=
    '/data/fxy_datasets/underwater_bs_deblur/test_annotations_clean.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        _scope_='mmdet',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet', max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        _scope_='mmdet',
        ann_file='train_annotations_clean.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='/data/fxy_datasets/underwater_bs_deblur/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='BSDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(_scope_='mmdet', keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
val_cfg = dict(type='mmrazor.SingleTeacherDistillValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='test_annotations_clean.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='/data/fxy_datasets/underwater_bs_deblur/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='BSDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file=
    '/data/fxy_datasets/underwater_bs_deblur/test_annotations_clean.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/distillation_uod'
