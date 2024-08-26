voxel_size = [0.075, 0.075, 0.2]

num_classes = 7 # 10 for detection, 7 for tracking
model = dict(
    type='JDT3D',
    data_preprocessor=dict(
        type='TrackDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    # lidar params
    voxelize_cfg=dict(
        max_num_points=10,
        # point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
        voxel_size=voxel_size,
        max_voxels=[120000, 160000]),
    voxelize_reduce=True,
    feat_channels=512,
    hidden_channel=256,
    # heatmap param
    heatmap_query_init=True,
    heatmap_nms_kernel=3,
    # PF-Track params
    tracking=False,
    train_backbone=True,
    use_grid_mask=True,
    if_update_ego=False, # update the ego-motion
    motion_prediction=False,
    motion_prediction_ref_update=False,
    num_query=500,
    num_classes=num_classes, # 10 for detection, 7 for tracking
    # pc_range=point_cloud_range,
    runtime_tracker=dict(
        output_threshold=0.0,
        score_threshold=0.0,
        record_threshold=0.0,
        max_age_since_update=1),
    spatial_temporal_reason=dict(
        history_reasoning=False,
        future_reasoning=False,
        hist_len=3,
        fut_len=4,
        num_classes=num_classes,
        # pc_range=point_cloud_range,
        hist_temporal_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        spatial_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        fut_temporal_transformer=dict(
            type='TemporalTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=2,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=5,
        sparse_shape=[1440, 1440, 41],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (1, 1, 0)), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='BEVFusionTrackingHead',
        num_classes=num_classes,
        in_channels=256,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
        num_pred=1, # num decoder layers
        transformer=dict(
            type='PETRTrackingTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=1,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='TrackNMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            # pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.0,
            num_classes=num_classes), 
        positional_encoding=dict(
            # type='mmdet.LearnedPositionalEncoding', num_feats=128, normalize=True),
            type='mmdet.SinePositionalEncoding', num_feats=128, normalize=True),
    ),
    # model training and testing settings
    test_cfg=dict(
        pts=None, # requured by mvx_two_stage
        grid_size=[1440, 1440, 41], # pc_range/voxel_size
        voxel_size=voxel_size,
        # point_cloud_range=point_cloud_range, # determined by dataset
        out_size_factor=8, # grid_size/feature_map_size
    ),
    loss=dict(
        type='TrackingLossCombo',
        num_classes=num_classes,
        interm_loss=True,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        gaussian_overlap=0.1,
        min_gauss_radius=2,
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0),
        loss_prediction=dict(type='mmdet.L1Loss', loss_weight=0.5),
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            # pc_range=point_cloud_range
        )
    )
)