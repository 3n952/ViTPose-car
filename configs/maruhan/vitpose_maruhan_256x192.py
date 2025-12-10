import os

data_root = os.getenv('MARUHAN_DATA_ROOT', '../Maruhan-car-kp')
benchmark_root = os.getenv('BENCHMARK_DATA_ROOT', 
                           '/home/skim/workspace/maruhan_demo/benchmark_dataset/Parking-.v1i.coco/train')

train_ann_file = os.path.join(data_root, 'train', '_annotations.coco.json')
# val_ann_file = os.path.join(data_root, 'val', '_annotations.coco.json')
# test_ann_file = os.path.join(data_root, 'test', '_annotations.coco.json')
benchmark_ann_file = os.path.join(benchmark_root, '_annotations.coco.json')

train_img_prefix = os.path.join(data_root, 'train') + '/'
# val_img_prefix = os.path.join(data_root, 'val') + '/'
# test_img_prefix = os.path.join(data_root, 'test') + '/'
benchmark_img_prefix = os.path.join(benchmark_root)

dataset_info = dict(
    dataset_name='maruhan_car_kp',
    paper_info=dict(
        author='Maruhan',
        title='Maruhan Car Keypoints',
        container='Internal',
        year='2025',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='front_right_wheel',
            id=0,
            color=[0, 255, 0],
            type='wheel',
            swap='front_left_wheel'),
        1:
        dict(
            name='rear_left_wheel',
            id=1,
            color=[255, 0, 0],
            type='wheel',
            swap='rear_right_wheel'),
        2:
        dict(
            name='rear_right_wheel',
            id=2,
            color=[0, 0, 255],
            type='wheel',
            swap='rear_left_wheel'),
        3:
        dict(
            name='front_left_wheel',
            id=3,
            color=[255, 255, 0],
            type='wheel',
            swap='front_right_wheel'),
        4:
        dict(
            name='front_left',
            id=4,
            color=[255, 128, 0],
            type='corner',
            swap='front_right'),
        5:
        dict(
            name='front_right',
            id=5,
            color=[0, 255, 255],
            type='corner',
            swap='front_left'),
        6:
        dict(
            name='rear_right',
            id=6,
            color=[128, 0, 255],
            type='corner',
            swap='rear_left'),
        7:
        dict(
            name='rear_left',
            id=7,
            color=[255, 0, 128],
            type='corner',
            swap='rear_right'),
        8:
        dict(
            name='top',
            id=8,
            color=[255, 255, 255],
            type='top',
            swap='top'),
    },
    skeleton_info={
        0:
        dict(
            link=('front_left_wheel', 'front_right_wheel'),
            id=0,
            color=[255, 255, 255]),
        1:
        dict(
            link=('front_right_wheel', 'rear_right_wheel'),
            id=1,
            color=[255, 255, 255]),
        2:
        dict(
            link=('rear_right_wheel', 'rear_left_wheel'),
            id=2,
            color=[255, 255, 255]),
        3:
        dict(
            link=('rear_left_wheel', 'front_left_wheel'),
            id=3,
            color=[255, 255, 255]),
        4:
        dict(link=('front_left', 'front_right'), id=4, color=[0, 255, 0]),
        5:
        dict(link=('rear_left', 'rear_right'), id=5, color=[0, 255, 0]),
        6:
        dict(link=('rear_right', 'rear_right_wheel'), id=6, color=[255, 0, 0]),
        7:
        dict(link=('rear_left', 'rear_left_wheel'), id=7, color=[255, 0, 0]),
        8:
        dict(link=('front_left', 'front_left_wheel'), id=8, color=[0, 0, 255]),
        9:
        dict(
            link=('front_right', 'front_right_wheel'),
            id=9,
            color=[0, 0, 255]),
        10:
        dict(link=('front_right', 'top'), id=10, color=[255, 255, 0]),
        11:
        dict(link=('front_left', 'top'), id=11, color=[255, 255, 0]),
        12:
        dict(link=('rear_left', 'top'), id=12, color=[255, 255, 0]),
        13:
        dict(link=('rear_right', 'top'), id=13, color=[255, 255, 0]),
    },
    joint_weights=[1.] * 9,
    sigmas=[0.05] * 9)

channel_cfg = dict(
    num_output_channels=9,
    dataset_joints=9,
    dataset_channel=[[0, 1, 2, 3, 4, 5, 6, 7, 8]],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8])

data_cfg = dict(
    image_size=[192, 256],  # TODO: change to [256, 192] for car crop
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0,
    bbox_file='')

target_type = 'GaussianHeatmap'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=30, scale_factor=0.35),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='TopDownGenerateTarget',
        sigma=2,
        encoding='UDP',
        target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=16),
    test_dataloader=dict(samples_per_gpu=16, workers_per_gpu=2),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info=dataset_info),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=train_ann_file,
        img_prefix=train_img_prefix,
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info),
    benchmark=dict(
        type='TopDownCocoDataset',
        ann_file=benchmark_ann_file,
        img_prefix=benchmark_img_prefix,
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info=dataset_info))

evaluation = dict(interval=10, metric='mAP', save_best='AP')

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys=dict(
            bias=dict(decay_mult=0.),
            pos_embed=dict(decay_mult=0.),
            relative_position_bias_table=dict(decay_mult=0.),
            norm=dict(decay_mult=0.))))
optimizer_config = dict(grad_clip=dict(max_norm=1., norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[100, 140])
total_epochs = 300

model = dict(
    type='TopDown',
    pretrained='checkpoints/ViTPose/vitpose-b.pth',
    backbone=dict(
        type='ViT',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True))

cudnn_benchmark = True
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
log_level = 'INFO'
checkpoint_config = dict(interval=10)
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
resume_from = None
load_from = None
work_dir = 'work_dirs/vitpose_maruhan'
