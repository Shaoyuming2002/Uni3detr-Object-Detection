
_base_ = [
    # '../../../configs/_base_/datasets/nus-3d.py',
    '../../../configs/_base_/schedules/cyclic-20e.py',
    '../../../configs/_base_/default_runtime.py'
]

custom_imports = dict(imports=['projects.Uni3DETR.uni3detr',
                               'projects.PETR.petr'
                    
                               ])
# # introduce mmdet3d plugin
# # 所以第一个改检查的就是plugin里面的文件，由于版本可能有很多问题
# plugin=True
# plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# voxelization for original point cloud
# pts_voxel_size = [0.15, 0.15, 0.3]
pts_voxel_size = [0.2, 0.2, 0.4]
# voxelization for bev image
voxel_size = [0.3, 0.3, 8]
# 使用多少帧激光雷达扫描
lidar_sweep_num = 10
#是否启用混合精度
# fp16_enabled = False
# bev下采样步长
bev_stride = 4
sample_num = 5

# 输入模态只用点云
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

"""
2. 模型结构: type='Uni3DETR'，包含多个子模块：
   - pts_voxel_layer: 体素化层，对原始点云进行离散化
   - pts_voxel_encoder: 将体素特征编码为稠密或稀疏表示
   - pts_middle_encoder: 通常是稀疏编码器(SparseEncoder)，将体素处理后得到主干输出
   - pts_backbone: 3D backbone (SECOND3D)
   - pts_neck: FPN之类的多尺度融合 (SECOND3DFPN)
   - pts_bbox_head: 核心检测头 (Uni3DETRHead) 
"""
model = dict(
    type='Uni3DETR',
    pts_voxel_layer=dict(
        type='VoxelizationByGridShape',
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=pts_voxel_size,
        max_voxels=(30000, 40000),
        deterministic=False,     # 是否使用确定性体素化
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        # sparse_shape=[41, 1440, 1440],    # 原来设个41
        sparse_shape=[21, 540, 540],
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        # encoder_channels=((16, 16, 16), (16, 16, 32), (32, 32, 64), (64, 128)),  
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        # encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type='basicblock',
        # fp16_enabled=False
        ), # not enable FP16 here
    # pts_backbone=dict(
    #     type='SECOND3D',
    #     in_channels=[256, 256, 256],
    #     out_channels=[128, 256, 512],
    #     layer_nums=[5, 5, 5],
    #     layer_strides=[1, 2, 4],
    #     is_cascade=False,
    #     norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
    #     conv_cfg=dict(type='Conv3d', kernel=(1,3,3), bias=False)),
    pts_neck=dict(
        type='SECOND3DFPN',
        in_channels=[128, 256, 512],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv3d', bias=False),
        extra_conv=dict(type='Conv3d', num_conv=3, bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='Uni3DETRHead',
        num_query=900, 
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='Uni3DETRTransformer',
            # fp16_enabled=fp16_enabled,
            decoder=dict(
                type='Uni3DETRTransformerDecoder',
                num_layers=3,    # decoder层数
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='UniCrossAtten',
                            num_points=1,
                            embed_dims=256,
                            num_sweeps=1,
                            # fp16_enabled=fp16_enabled
                            )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    norm_cfg=dict(type='LN'),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=900,
            # alpha=1.0,
            voxel_size=voxel_size,
            num_classes=10), 
        post_processing=dict(
            type='nms',
            nms_thr=0.2,
            num_thr=500),
        positional_encoding=dict(
            type='SinePositionalEncoding3D',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(type='SoftFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='IoU3DLoss', loss_weight=1.2),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        train_cfg=dict(
    grid_size=[720, 720, 1],
    voxel_size=voxel_size,
    point_cloud_range=point_cloud_range,
    out_size_factor=bev_stride,
    assigner=dict(
        type='HungarianAssigner3D1',
        cls_cost=dict(type='FocalLossCost', weight=2.0),
        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
        iou_cost=dict(type='IoU3DCost', weight=1.2),
        pc_range=point_cloud_range))))
    # model training and testing settings
    # train_cfg=dict(
    #     pts=dict(
    #     grid_size=[720, 720, 1],
    #     voxel_size=voxel_size,
    #     point_cloud_range=point_cloud_range,
    #     out_size_factor=bev_stride,
    #     assigner=dict(
    #         type='HungarianAssigner3D1',
    #         cls_cost=dict(type='FocalLossCost', weight=2.0),
    #         reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
    #         iou_cost=dict(type='IoU3DCost', weight=1.2),
    #         pc_range=point_cloud_range))))
    # train_cfg=dict(
    # grid_size=[720, 720, 1],
    # voxel_size=voxel_size,
    # point_cloud_range=point_cloud_range,
    # out_size_factor=bev_stride,
    # assigner=dict(
    #     type='HungarianAssigner3D1',
    #     cls_cost=dict(type='FocalLossCost', weight=2.0),
    #     reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
    #     iou_cost=dict(type='IoU3DCost', weight=1.2),
    #     pc_range=point_cloud_range)))


# dataset_type = 'NuScenesSweepDataset'
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)

# backend_args = dict(_delete_=True)
backend_args = {}


db_sampler = dict(
    type='DataBaseSampler',
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl', 
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        backend_args=backend_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=lidar_sweep_num-1,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectSample', db_sampler=db_sampler), # commit this for the last 4 epoch
    dict(
        # type='UnifiedRotScaleTrans',
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        # type='UnifiedRandomFlip3D',
        type='RandomFlip3D',
        # sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'points'])
    dict(type='Pack3DDetInputs', keys=['gt_bboxes_3d', 'gt_labels_3d', 'points'], meta_keys=['img_metas', 'fpsbpts'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=lidar_sweep_num-1,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='CollectUnified3D', keys=['points'])
    dict(type='Pack3DDetInputs', keys=['gt_bboxes_3d', 'gt_labels_3d', 'points'], meta_keys=['img_metas', 'fpsbpts'])
    # dict(
    #     type='MultiScaleFlipAug3D',
    #     img_scale=(1333, 800),
    #     pts_scale_ratio=1,
    #     # Add double-flip augmentation
    #     flip=True,
    #     pcd_horizontal_flip=True,
    #     pcd_vertical_flip=True,
    #     transforms=[
    #         dict(
    #             type='GlobalRotScaleTrans',
    #             rot_range=[0, 0],
    #             scale_ratio_range=[1., 1.],
    #             translation_std=[0, 0, 0]),
    #         dict(type='RandomFlip3D', sync_2d=False),
    #         dict(
    #             type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    #         dict(
    #             type='DefaultFormatBundle3D',
    #             class_names=class_names,
    #             with_label=False),
    #         dict(type='Collect3D', keys=['points'])
    #     ])
]


# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=4,
#     train_dataloader=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=data_root,
#             ann_file=data_root + 'nuscenes_infos_train.pkl',
#             pipeline=train_pipeline,
#             classes=class_names,
#             modality=input_modality,
#             test_mode=False,
#             use_valid_flag=True,
#             box_type_3d='LiDAR')),
#     val_dataloader=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'nuscenes_infos_val.pkl',
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'),
#     test_dataloader=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'nuscenes_infos_val.pkl',
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'))
# samples_per_gpu=4,
# workers_per_gpu=4,
train_dataloader=dict(
        # samples_per_gpu=4,
        # workers_per_gpu=4,
        dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_prefix=dict(
                pts='samples/LIDAR_TOP',
                sweeps='sweeps/LIDAR_TOP'),
            data_root=data_root,
            ann_file='nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            # classes=class_names,
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')))
val_dataloader=dict(
    dataset=dict(
    type=dataset_type,
    data_prefix=dict(
        pts='samples/LIDAR_TOP',
        sweeps='sweeps/LIDAR_TOP'),
    data_root=data_root,
    ann_file='nuscenes_infos_val.pkl',
    pipeline=test_pipeline,
    # classes=class_names,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR'))
test_dataloader=dict(
    dataset=dict(
    type=dataset_type,
    data_prefix=dict(
        pts='samples/LIDAR_TOP',
        sweeps='sweeps/LIDAR_TOP'),
    data_root=data_root,
    ann_file='nuscenes_infos_test.pkl',
    pipeline=test_pipeline,
    # classes=class_names,
    metainfo=metainfo,
    modality=input_modality,
    test_mode=True,
    box_type_3d='LiDAR'))


# evaluation = dict(interval=4, pipeline=test_pipeline)
# val_evaluator = dict(type='NuScenesEvaluator', interval=4, pipeline=test_pipeline)
# test_evaluator = dict(type='NuScenesEvaluator', interval=4, pipeline=test_pipeline)
# val_evaluator = dict(type='CBGSEvaluator', interval=4, pipeline=test_pipeline)
# test_evaluator = dict(type='CBGSEvaluator', interval=4, pipeline=test_pipeline)

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file='nuscenes_infos_val.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=20)

# train_cfg = dict(max_epochs=20, val_interval=20)
find_unused_parameters = True
# fp16 setting
# fp16 = dict(loss_scale=32.)