# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import numpy as np
import torch
from mmdet3d.datasets.transforms.transforms_3d import (
    ObjectNameFilter,
    ObjectRangeFilter,
    PointsRangeFilter,
    ObjectSample,
)
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
)
from PIL import Image

from projects.BEVFusion.bevfusion.transforms_3d import (
    BEVFusionGlobalRotScaleTrans,
    BEVFusionRandomFlip3D,
)
from projects.PETR.petr.transforms_3d import (
    GlobalRotScaleTransImage,
    ResizeCropFlipImage,
)


@TRANSFORMS.register_module()
class TrackInstanceRangeFilter(ObjectRangeFilter):
    def transform(self, input_dict: dict) -> dict:
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            input_dict["gt_bboxes_3d"], (LiDARInstance3DBoxes, DepthInstance3DBoxes)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]
        else:
            raise TypeError(
                f"Invalid points instance type " f'{type(input_dict["gt_bboxes_3d"])}'
            )

        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        instance_inds = input_dict["instance_inds"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(bool)]
        instance_inds = instance_inds[mask.numpy().astype(bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        input_dict["instance_inds"] = instance_inds

        # hacks for forecasting
        if "gt_forecasting_locs" in input_dict.keys():
            input_dict["gt_forecasting_locs"] = input_dict["gt_forecasting_locs"][
                mask.numpy().astype(bool)
            ]
            input_dict["gt_forecasting_masks"] = input_dict["gt_forecasting_masks"][
                mask.numpy().astype(bool)
            ]
            input_dict["gt_forecasting_types"] = input_dict["gt_forecasting_types"][
                mask.numpy().astype(bool)
            ]

        return input_dict


@TRANSFORMS.register_module()
class SeqTrackInstanceRangeFilter(TrackInstanceRangeFilter):
    # TODO switch to using TransformWrapper
    def transform(self, data_queue: list) -> list:
        transformed_queue = []
        for single_result in data_queue:
            transformed_queue.append(self.transform_single_result(single_result))
        return transformed_queue

    def transform_single_result(self, input_dict: dict) -> dict:
        return super().transform(input_dict)


@TRANSFORMS.register_module()
class TrackObjectNameFilter(ObjectNameFilter):
    def transform(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d], dtype=bool)
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]

        # hacks for forecasting
        if "gt_forecasting_locs" in input_dict.keys():
            input_dict["gt_forecasting_locs"] = input_dict["gt_forecasting_locs"][
                gt_bboxes_mask
            ]
            input_dict["gt_forecasting_masks"] = input_dict["gt_forecasting_masks"][
                gt_bboxes_mask
            ]
            input_dict["gt_forecasting_types"] = input_dict["gt_forecasting_types"][
                gt_bboxes_mask
            ]

        return input_dict


@TRANSFORMS.register_module()
class TrackResizeCropFlipImage(ResizeCropFlipImage):
    def transform(self, data_queue: list):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        for frame_idx, data in enumerate(data_queue):
            new_imgs_frame = []
            transforms = []
            for cam_idx, img in enumerate(data["img"]):
                img = Image.fromarray(np.uint8(img))
                # augmentation (resize, crop, horizontal flip, rotate)
                img, ida_mat = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                )
                new_imgs_frame.append(np.array(img).astype(np.float32))
                data_queue[frame_idx]["intrinsics"][cam_idx][:3, :3] = (
                    ida_mat @ data_queue[frame_idx]["intrinsics"][cam_idx][:3, :3]
                )
                # convert from 3x4 to 4x4 matrix for BEVFusion DepthLSS
                transform = torch.eye(4)
                transform[:2, :2] = ida_mat[:2, :2]
                transform[:2, 3] = ida_mat[:2, 2]
                transforms.append(transform.numpy())

            data_queue[frame_idx]["img"] = new_imgs_frame
            data_queue[frame_idx]["lidar2img"] = [
                intrinsics @ extrinsics.T
                for intrinsics, extrinsics in zip(
                    data_queue[frame_idx]["intrinsics"],
                    data_queue[frame_idx]["extrinsics"],
                )
            ]
            data_queue[frame_idx]["img_aug_matrix"] = transforms

        return data_queue


@TRANSFORMS.register_module()
class TrackGlobalRotScaleTransImage(GlobalRotScaleTransImage):
    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline. Containing multiple frames.
        Returns:
            dict: Updated result dict.
        """
        frame_num = len(results["timestamp"])

        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle, frame_num)
        if self.reverse_angle:
            rot_angle *= -1

        for i in range(frame_num):
            results["gt_bboxes_3d"][i].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio, frame_num)
        for i in range(frame_num):
            results["gt_bboxes_3d"][i].scale(scale_ratio)

        # TODO: support translation

        return results

    def rotate_bev_along_z(self, results, angle, frame_num):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for i in range(frame_num):
            for view in range(num_view):
                results["lidar2img"][i][view] = (
                    torch.tensor(results["lidar2img"][i][view]).float() @ rot_mat_inv
                ).numpy()

        return

    def scale_xyz(self, results, scale_ratio, frame_num):
        rot_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for i in range(frame_num):
            for view in range(num_view):
                results["lidar2img"][i][view] = (
                    torch.tensor(results["lidar2img"][i][view]).float() @ rot_mat_inv
                ).numpy()

        return


@TRANSFORMS.register_module()
class SeqBEVFusionGlobalRotScaleTrans(BEVFusionGlobalRotScaleTrans):
    """Compared with `BEVFusionGlobalRotScaleTrans`, this transform takes
    in a sequence of frames and applies the same augmentation to each of them."""

    def transform(self, data_queue: list) -> list:
        # determine augmentation parameters for the clip
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])
        scale_factor = np.random.uniform(
            self.scale_ratio_range[0], self.scale_ratio_range[1]
        )
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        transformed_queue = []
        for single_result in data_queue:
            transformed_queue.append(
                self.transform_single_result(
                    single_result, noise_rotation, scale_factor, trans_factor
                )
            )
        return transformed_queue

    def transform_single_result(
        self, input_dict, noise_rotation, scale_factor, trans_factor
    ):
        """Based on BEVFusionGlobalRotScaleTrans.transform()

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling. The keys 'points', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans' and `gt_bboxes_3d` are updated
            in the result dict."""
        if "transformation_3d_flow" not in input_dict:
            input_dict["transformation_3d_flow"] = []

        self._rot_bbox_points(input_dict, noise_rotation)

        if "pcd_scale_factor" not in input_dict:
            input_dict["pcd_scale_factor"] = scale_factor
        self._trans_bbox_points(input_dict, trans_factor)
        self._scale_bbox_points(input_dict)

        input_dict["transformation_3d_flow"].extend(["R", "T", "S"])

        lidar_augs = np.eye(4)
        lidar_augs[:3, :3] = (
            input_dict["pcd_rotation"].T * input_dict["pcd_scale_factor"]
        )
        lidar_augs[:3, 3] = input_dict["pcd_trans"] * input_dict["pcd_scale_factor"]

        if "gt_forecasting_locs" in input_dict.keys():
            augmented_locs = input_dict["gt_forecasting_locs"].copy()
            augmented_locs = np.concatenate(
                [augmented_locs, np.ones((*augmented_locs.shape[:-1], 1))], axis=-1
            )
            input_dict["gt_forecasting_locs"] = augmented_locs @ lidar_augs[:3, :].T

        if "lidar_aug_matrix" not in input_dict:
            input_dict["lidar_aug_matrix"] = np.eye(4)
        input_dict["lidar_aug_matrix"] = lidar_augs @ input_dict["lidar_aug_matrix"]

        return input_dict

    def _rot_bbox_points(self, input_dict: dict, noise_rotation: float) -> None:
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            noise_rotation (float): Rotation angle.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
            and `gt_bboxes_3d` is updated in the result dict.
        """

        if "gt_bboxes_3d" in input_dict and len(input_dict["gt_bboxes_3d"].tensor) != 0:
            # rotate points with bboxes
            points, rot_mat_T = input_dict["gt_bboxes_3d"].rotate(
                noise_rotation, input_dict["points"]
            )
            input_dict["points"] = points
        else:
            # if no bbox in input_dict, only rotate points
            rot_mat_T = input_dict["points"].rotate(noise_rotation)

        input_dict["pcd_rotation"] = rot_mat_T
        input_dict["pcd_rotation_angle"] = noise_rotation

    def _trans_bbox_points(self, input_dict: dict, trans_factor: float) -> None:
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            trans_factor (float): Translation factor.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
            and `gt_bboxes_3d` is updated in the result dict.
        """
        input_dict["points"].translate(trans_factor)
        input_dict["pcd_trans"] = trans_factor
        if "gt_bboxes_3d" in input_dict:
            input_dict["gt_bboxes_3d"].translate(trans_factor)


@TRANSFORMS.register_module()
class SeqBEVFusionRandomFlip3D(BEVFusionRandomFlip3D):
    """Compared with `BEVFusionRandomFlip3D`, this transform takes
    in a sequence of frames and applies the same flip augmentation to each of them."""

    def transform(self, data_queue: list) -> list:
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])
        transformed_queue = []
        for single_result in data_queue:
            transformed_queue.append(
                self.transform_single_result(
                    single_result, flip_horizontal, flip_vertical
                )
            )
        return transformed_queue

    def transform_single_result(
        self, data: dict, flip_horizontal, flip_vertical
    ) -> dict:
        """Based on BEVFusionRandomFlip3D.__call__()"""
        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("horizontal")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("horizontal")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, :, ::-1].copy()
            if "gt_forecasting_locs" in data:  # [num objs, num forecasting locs, 3]
                data["gt_forecasting_locs"][..., 1] *= -1

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if "points" in data:
                data["points"].flip("vertical")
            if "gt_bboxes_3d" in data:
                data["gt_bboxes_3d"].flip("vertical")
            if "gt_masks_bev" in data:
                data["gt_masks_bev"] = data["gt_masks_bev"][:, ::-1, :].copy()
            if "gt_forecasting_locs" in data:
                data["gt_forecasting_locs"][..., 0] *= -1

        if "lidar_aug_matrix" not in data:
            data["lidar_aug_matrix"] = np.eye(4)
        data["lidar_aug_matrix"][:3, :] = rotation @ data["lidar_aug_matrix"][:3, :]
        return data


@TRANSFORMS.register_module()
class SeqPointsRangeFilter(PointsRangeFilter):
    # TODO switch to using TransformWrapper
    def transform(self, data_queue: list) -> list:
        transformed_queue = []
        for single_result in data_queue:
            transformed_queue.append(self.transform_single_result(single_result))
        return transformed_queue

    def transform_single_result(self, input_dict: dict) -> dict:
        return super().transform(input_dict)


@TRANSFORMS.register_module()
class TrackSample(ObjectSample):
    """
    TODO: support 2D sampling
    TODO: support ground plane sampling
    """

    def transform(self, data_queue: list) -> list:
        if self.disabled:
            return data_queue
        # sample tracks from the db
        sampled_dicts = self.db_sampler.sample_all(data_queue)
        # inject tracks into each frame of the data queue
        for i, (input_dict, sampled_dict) in enumerate(zip(data_queue, sampled_dicts)):
            data_queue[i] = self.transform_single_result(input_dict, sampled_dict)
        return data_queue

    def transform_single_result(self, input_dict: dict, sampled_dict: dict) -> dict:
        if sampled_dict is None:
            return input_dict
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_instance_inds = input_dict["instance_inds"]
        gt_forecasting_locs = input_dict["gt_forecasting_locs"]
        gt_forecasting_masks = input_dict["gt_forecasting_masks"]
        gt_forecasting_types = input_dict["gt_forecasting_types"]
        points = input_dict["points"]

        sampled_gt_bboxes_3d = sampled_dict["gt_bboxes_3d"]
        sampled_gt_labels = sampled_dict["gt_labels_3d"]
        sampled_gt_instance_inds = sampled_dict["instance_inds"]
        sampled_gt_forecasting_locs = sampled_dict["gt_forecasting_locs"]
        sampled_gt_forecasting_masks = sampled_dict["gt_forecasting_masks"]
        sampled_gt_forecasting_types = sampled_dict["gt_forecasting_types"]
        sampled_points = sampled_dict["points"]

        # add sampled annotations to the original gt annotations
        gt_labels_3d = np.concatenate(
            [gt_labels_3d, sampled_gt_labels], axis=0, dtype=np.int64
        )
        gt_bboxes_3d = gt_bboxes_3d.new_box(
            np.concatenate([gt_bboxes_3d.numpy(), sampled_gt_bboxes_3d])
        )
        gt_instance_inds = np.concatenate(
            [gt_instance_inds, sampled_gt_instance_inds], axis=0
        )
        gt_forecasting_locs = np.concatenate(
            [gt_forecasting_locs, sampled_gt_forecasting_locs], axis=0
        )
        gt_forecasting_masks = np.concatenate(
            [gt_forecasting_masks, sampled_gt_forecasting_masks], axis=0
        )
        gt_forecasting_types = np.concatenate(
            [gt_forecasting_types, sampled_gt_forecasting_types], axis=0
        )

        # insert the sampled points into the points tensor
        points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
        points = points.cat([sampled_points, points])

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        input_dict["points"] = points
        input_dict["instance_inds"] = gt_instance_inds
        input_dict["gt_forecasting_locs"] = gt_forecasting_locs
        input_dict["gt_forecasting_masks"] = gt_forecasting_masks
        input_dict["gt_forecasting_types"] = gt_forecasting_types

        return input_dict
