_base_ = [
    "./jdt3d_f1.py",
]

train_pipeline = [
    dict(
        type="mmdet3d.LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        backend_args={{_base_.backend_args}},
    ),
    dict(
        type="mmdet3d.LoadPointsFromMultiSweeps",
        sweeps_num=10,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args={{_base_.backend_args}},
    ),
    dict(
        type="mmdet3d.TrackLoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_forecasting=True,
    ),
    dict(type="mmdet3d.TrackObjectNameFilter", classes={{_base_.class_names}}),
    dict(type="mmdet3d.PointShuffle"),
]

train_pipeline_multiframe = [
    dict(type='mmdet3d.TrackSample', db_sampler={{_base_.db_sampler}}),
    dict(
        type='mmdet3d.SeqBEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='mmdet3d.SeqBEVFusionRandomFlip3D'),
    dict(
        type="mmdet3d.SeqPointsRangeFilter",
        point_cloud_range={{_base_.point_cloud_range}},
    ),
    dict(
        type="mmdet3d.SeqTrackInstanceRangeFilter",
        point_cloud_range={{_base_.point_cloud_range}},
    ),
    dict(
        type="mmdet3d.Pack3DTrackInputs",
        keys=[
            "points",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "instance_inds",
            "gt_forecasting_locs",
            "gt_forecasting_masks",
            "gt_forecasting_types",
        ],
        meta_keys=[
            "cam2img",
            "ori_cam2img",
            "lidar2cam",
            "lidar2img",
            "cam2lidar",
            "ori_lidar2img",
            "img_aug_matrix",
            "box_type_3d",
            "sample_idx",
            "lidar_path",
            "lidar2global",
            "img_path",
            "transformation_3d_flow",
            "pcd_rotation",
            "pcd_scale_factor",
            "pcd_trans",
            "img_aug_matrix",
            "lidar_aug_matrix",
            "num_pts_feats",
            "timestamp",
            "pad_shape",
        ],
    ),
]

# turn off cbgs to speed up training
num_frames_per_sample = 3
# without CBGS
train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        num_frames_per_sample=num_frames_per_sample,
        forecasting=True,
        data_root={{_base_.data_root}},
        ann_file={{_base_.train_pkl_path}},
        pipeline=train_pipeline,
        pipeline_multiframe=train_pipeline_multiframe,
        metainfo={{_base_.metainfo}},
        modality={{_base_.input_modality}},
        test_mode=False,
        data_prefix={{_base_.data_prefix}},
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        use_valid_flag=True,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        backend_args={{_base_.backend_args}},
    )
)

# With CBGS
# train_dataloader = dict(
#     dataset=dict(
#         dataset=dict(
#             num_frames_per_sample=num_frames_per_sample,
#             pipeline=train_pipeline,
#         )
#     )
# )

# when evaluating NuScenesTrackingMetric, we need to use a special dataloader sampler
val_dataloader = dict(
    sampler=dict(type="TrackSampler3D", _delete_=True,),
)
test_dataloader = dict(
    sampler=dict(type="TrackSampler3D", _delete_=True,),
)

model = dict(
    tracking=True,
    train_backbone=True,
    if_update_ego=True,
    motion_prediction=True,
    motion_prediction_ref_update=True,
    runtime_tracker=dict(
        output_threshold=0.2,
        score_threshold=0.4,
        record_threshold=0.4,
        max_age_since_update=7,
    ),
    spatial_temporal_reason=dict(
        history_reasoning=True,
        future_reasoning=True,
        fut_len=8,
    ),
    pts_bbox_head=dict(
        bbox_coder=dict(
            max_num=150,
        ),
    ),
)

val_evaluator = dict(
    type="NuScenesTrackingMetric", jsonfile_prefix="work_dirs/nuscenes_results/tracking"
)
test_evaluator = dict(
    type="NuScenesTrackingMetric", jsonfile_prefix="work_dirs/nuscenes_results/tracking"
)
load_from = 'work_dirs/tracking/lidar/f1/2023-11-02/epoch_16.pth'
