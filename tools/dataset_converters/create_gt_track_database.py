# Based on tools/dataset_converters/create_gt_database.py
import pickle
from os import path as osp

import mmengine
import numpy as np
from mmdet3d.registry import DATASETS
from mmdet3d.structures.ops import box_np_ops as box_np_ops
from mmengine import track_iter_progress

from projects.tracking_plugin import *


def create_groundtruth_track_database(
    dataset_class_name,
    data_path,
    info_prefix,
    info_path=None,
    mask_anno_path=None,
    used_classes=None,
    database_save_path=None,
    db_info_save_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    with_mask=False,
):
    """Given the raw data, generate the ground truth database.
    The database contains track trajectories all together

    Currently only support NuScenes dataset.
    # TODO support other datasets

    Args:
        dataset_class_name (str): Name of the input dataset.
        data_path (str): Path of the data.
        info_prefix (str): Prefix of the info file.
        info_path (str, optional): Path of the info file.
            Default: None.
    """
    print(f"Create GT Database of {dataset_class_name}")
    assert dataset_class_name in [
        "NuScenesTrackingDataset"
    ], "Only support NuScenesTrackingDataset for now"
    dataset_cfg = dict(
        type=dataset_class_name,
        data_root=data_path,
        ann_file=info_path,
        use_valid_flag=True,
        data_prefix=dict(pts="samples/LIDAR_TOP", img="", sweeps="sweeps/LIDAR_TOP"),
        pipeline=[
            dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=5),
            dict(
                type="LoadPointsFromMultiSweeps",
                sweeps_num=10,
                use_dim=[0, 1, 2, 3, 4],
                pad_empty_sweeps=True,
                remove_close=True,
            ),
            dict(
                type="TrackLoadAnnotations3D",
                with_bbox_3d=True,
                with_label_3d=True,
                with_forecasting=True,
            ),
        ],
    )

    dataset = DATASETS.build(dataset_cfg)

    if database_save_path is None:
        database_save_path = osp.join(data_path, f"{info_prefix}_track_gt_database")
    if db_info_save_path is None:
        db_info_save_path = osp.join(data_path, f"{info_prefix}_track_dbinfos_train.pkl")
    mmengine.mkdir_or_exist(database_save_path)
    all_db_infos = dict()
    # if with_mask:
    #     coco = COCO(osp.join(data_path, mask_anno_path))
    #     imgIds = coco.getImgIds()
    #     file2id = dict()
    #     for i in imgIds:
    #         info = coco.loadImgs([i])[0]
    #         file2id.update({info['file_name']: i})

    group_counter = 0
    for j in track_iter_progress(list(range(len(dataset)))):
        data_info = dataset.get_data_info(j)
        if len(data_info["ann_info"]['gt_labels_3d']) == 0:
            continue
        example = dataset.pipeline(data_info)
        annos = example["ann_info"]
        image_idx = example["sample_idx"]
        points = example["points"].numpy()
        gt_boxes_3d = annos["gt_bboxes_3d"].numpy()
        gt_labels = [dataset.metainfo["classes"][i] for i in annos["gt_labels_3d"]]
        instance_inds = annos["instance_inds"]
        group_dict = dict()
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes_3d.shape[0], dtype=np.int64)
        difficulty = np.zeros(gt_boxes_3d.shape[0], dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]

        num_obj = gt_boxes_3d.shape[0]
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes_3d)

        # if with_mask:
        #     # prepare masks
        #     gt_boxes = annos["gt_bboxes"]
        #     img_path = osp.split(example["img_info"]["filename"])[-1]
        #     if img_path not in file2id.keys():
        #         print(f"skip image {img_path} for empty mask")
        #         continue
        #     img_id = file2id[img_path]
        #     kins_annIds = coco.getAnnIds(imgIds=img_id)
        #     kins_raw_info = coco.loadAnns(kins_annIds)
        #     kins_ann_info = _parse_coco_ann_info(kins_raw_info)
        #     h, w = annos["img_shape"][:2]
        #     gt_masks = [_poly2mask(mask, h, w) for mask in kins_ann_info["masks"]]
        #     # get mask inds based on iou mapping
        #     bbox_iou = bbox_overlaps(kins_ann_info["bboxes"], gt_boxes)
        #     mask_inds = bbox_iou.argmax(axis=0)
        #     valid_inds = bbox_iou.max(axis=0) > 0.5

        #     # mask the image
        #     # use more precise crop when it is ready
        #     # object_img_patches = np.ascontiguousarray(
        #     #     np.stack(object_img_patches, axis=0).transpose(0, 3, 1, 2))
        #     # crop image patches using roi_align
        #     # object_img_patches = crop_image_patch_v2(
        #     #     torch.Tensor(gt_boxes),
        #     #     torch.Tensor(mask_inds).long(), object_img_patches)
        #     object_img_patches, object_masks = crop_image_patch(
        #         gt_boxes, gt_masks, mask_inds, annos["img"]
        #     )

        for i in range(num_obj):
            filename = f"{image_idx}_{gt_labels[i]}_{i}.bin"
            abs_filepath = osp.join(database_save_path, filename)
            rel_filepath = osp.join(f"{info_prefix}_track_gt_database", filename)

            # save point clouds and image patches for each object
            gt_points = points[point_indices[:, i]]
            gt_points[:, :3] -= gt_boxes_3d[i, :3]  # center the points about the object

            # if with_mask:
            #     if object_masks[i].sum() == 0 or not valid_inds[i]:
            #         # Skip object for empty or invalid mask
            #         continue
            #     img_patch_path = abs_filepath + ".png"
            #     mask_patch_path = abs_filepath + ".mask.png"
            #     mmcv.imwrite(object_img_patches[i], img_patch_path)
            #     mmcv.imwrite(object_masks[i], mask_patch_path)

            with open(abs_filepath, "w") as f:
                gt_points.tofile(f)

            if (used_classes is None) or gt_labels[i] in used_classes:
                db_info = {
                    "name": gt_labels[i],
                    "path": rel_filepath,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes_3d[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i],
                    "valid": True,
                    "instance_ind": instance_inds[i],
                    "forecasting_locs": annos["forecasting_locs"][
                        i
                    ],  # [13 locs, 3 dims]
                    "forecasting_masks": annos["forecasting_masks"][
                        i
                    ],  # [13 locs, 1 dim]
                    "forecasting_types": annos["forecasting_types"][
                        i
                    ],  # [13 locs, 1 dim]
                }
                local_group_id = group_ids[i]
                # if local_group_id >= 0:
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                # if with_mask:
                #     db_info.update({"box2d_camera": gt_boxes[i]})
                if gt_labels[i] not in all_db_infos:
                    all_db_infos[gt_labels[i]] = {}
                if instance_inds[i] not in all_db_infos[gt_labels[i]]:
                    all_db_infos[gt_labels[i]][instance_inds[i]] = []
                all_db_infos[gt_labels[i]][instance_inds[i]].append(db_info)

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, "wb") as f:
        pickle.dump(all_db_infos, f)
