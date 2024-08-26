# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
from typing import List, Union

import numpy as np
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.registry import DATASETS
from mmengine.dataset import Compose
from mmengine.logging import print_log


@DATASETS.register_module()
class NuScenesTrackingDataset(NuScenesDataset):
    CLASSES = ('car', 'truck', 'bus', 'trailer', 
               'motorcycle', 'bicycle', 'pedestrian', 
               'construction_vehicle', 'traffic_cone', 'barrier')
    def __init__(self,
                 pipeline_multiframe=None,
                 num_frames_per_sample=2,
                 forecasting=False,
                 ratio=1,
                 *args, **kwargs,
                 ):
        self.num_frames_per_sample = num_frames_per_sample
        self.pipeline_multiframe = pipeline_multiframe
        self.forecasting = forecasting
        self.ratio = ratio # divide the samples by a certain ratio, useful for quicker ablations
        if self.pipeline_multiframe is not None:
            self.pipeline_multiframe = Compose(self.pipeline_multiframe)
        self.scene_tokens = []
        self.indices_per_scene = {}
        self.num_frames_per_scene = {}
        super().__init__(*args, **kwargs)
    
    def __len__(self):
        if not self.test_mode:
            return super().__len__() // self.ratio
        else:
            return super().__len__()

    def prepare_data(self, index: int) -> Union[dict, None]:
        input_dict = self._prepare_data_single(index)
        if input_dict is None:
            print("input dict is None")
            print("index is: ", index)
            return None
        ann_info = input_dict['ann_info'] if not self.test_mode \
            else input_dict['eval_ann_info']
        if self.filter_empty_gt and (~(ann_info['gt_labels_3d'] != -1).any()):
            print("self.filter_empty_gt and (~(ann_info['gt_labels_3d'] != -1).any())")
            print('self.filter_empty_gt', self.filter_empty_gt)
            print(ann_info['gt_labels_3d'])
            print("index is: ", index)
            return None
        scene_token = input_dict['scene_token']
        data_queue = [input_dict]

        index_list = self.generate_track_data_indexes(index)
        index_list = index_list[::-1]
        for i in index_list[1:]:
            data_info_i = self._prepare_data_single(i)
            if data_info_i is None or data_info_i['scene_token'] != scene_token:
                print("data_info_i is None or data_info_i['scene_token'] != scene_token")
                print("data_info_i is None: ", data_info_i is None)
                if data_info_i is not None:
                    print("data_info_i['scene_token'] != scene_token: ", data_info_i['scene_token'] != scene_token)
                print("index is: ", i)
                return None
            ann_info = data_info_i['ann_info'] if not self.test_mode \
                else data_info_i['eval_ann_info']
            if self.filter_empty_gt and ~(ann_info['gt_labels_3d'] != -1).any() and not self.test_mode:
                print("self.filter_empty_gt and ~(ann_info['gt_labels_3d'] != -1).any() and not self.test_mode")
                print('self.filter_empty_gt', self.filter_empty_gt)
                print('~(ann_info[\'gt_labels_3d\'] != -1).any()', ~(ann_info['gt_labels_3d'] != -1).any())
                print('not self.test_mode', not self.test_mode)
                print("index is: ", i)
                return None
            data_queue.append(data_info_i)

        # return to the normal frame order
        data_queue = data_queue[::-1]

        # multiframe processing
        data = self.pipeline_multiframe(data_queue)
        return data

    def _prepare_data_single(self, index: int) -> Union[dict, None]:
        """Data preparation for both training and testing stage.

        Called by `__getitem__`  of dataset.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict or None: Data dict of the corresponding index.
        """
        ori_input_dict = self.get_data_info(index)

        # deepcopy here to avoid inplace modification in pipeline.
        input_dict = copy.deepcopy(ori_input_dict)

        # box_type_3d (str): 3D box type.
        input_dict['box_type_3d'] = self.box_type_3d
        # box_mode_3d (str): 3D box mode.
        input_dict['box_mode_3d'] = self.box_mode_3d

        # pre-pipline return None to random another in `__getitem__`
        if not self.test_mode and self.filter_empty_gt:
            if len(input_dict['ann_info']['gt_labels_3d']) == 0:
                print("filter empty gt in prepare data")
                print("index is: ", index)
                return None

        example = self.pipeline(input_dict)

        if not self.test_mode and self.filter_empty_gt:
            # after pipeline drop the example with empty annotations
            # return None to random another in `__getitem__`
            if example is None or len(
                    example['gt_bboxes_3d']) == 0:
                print("example is None or len(example['gt_bboxes_3d']) == 0")
                print("example is None: ", example is None)
                print("len(example['gt_bboxes_3d']) == 0: ", len(example['gt_bboxes_3d']) == 0)
                print('index is: ', index)
                return None

        if self.show_ins_var:
            if 'ann_info' in ori_input_dict:
                self._show_ins_var(
                    ori_input_dict['ann_info']['gt_labels_3d'],
                    example['gt_bboxes_3d'])
            else:
                print_log(
                    "'ann_info' is not in the input dict. It's probably that "
                    'the data is not in training mode',
                    'current',
                    level=30)

        return example

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        data_info = super().parse_data_info(info)
        scene_token = data_info['scene_token']
        self.scene_tokens.append(scene_token)
        self.indices_per_scene[scene_token] = self.indices_per_scene.get(
            scene_token, []) + [len(self.scene_tokens) - 1]
        # ego movement represented by lidar2global
        l2e = np.array(data_info['lidar_points']['lidar2ego'])
        e2g = np.array(data_info['ego2global'])
        l2g = e2g @ l2e

        # points @ R.T + T
        data_info.update(lidar2global=l2g.astype(np.float32))

        if self.modality['use_camera']:
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['images'].items():
                # obtain lidar to image transformation matrix
                lidar2cam_rt = np.array(cam_info['lidar2cam']).T
                intrinsic = np.array(cam_info['cam2img'])
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt) # the transpose of extrinsic matrix

            data_info.update(
                dict(
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                ))
        return data_info

    def generate_track_data_indexes(self, index):
        """Choose the track indexes that are within the same sequence
        """
        index_list = [i for i in range(index - self.num_frames_per_sample + 1, index + 1)]
        scene_tokens = [self.scene_tokens[i] for i in index_list]
        tgt_scene_token, earliest_index = scene_tokens[-1], index_list[-1]
        for i in range(self.num_frames_per_sample)[::-1]:
            if scene_tokens[i] == tgt_scene_token:
                earliest_index = index_list[i]
            elif self.test_mode:
                index_list = index_list[i + 1:]
                break
            elif (not self.test_mode):
                index_list[i] = earliest_index
        return index_list

    def get_num_scenes(self):
        return len(self.num_frames_per_scene)
    
    def get_scene_token(self, index):
        return self.scene_tokens[index]

    def get_all_scene_tokens(self):
        return sorted(list(set(self.scene_tokens)))