from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet3d.registry import MODELS
from mmengine.structures import InstanceData
from torch.nn import functional as F

from projects.BEVFusion.bevfusion.ops import Voxelization
from projects.PETR.petr.petr_head import pos2posemb3d
from projects.tracking_plugin.core.instances import Instances

from .Cam3DTracker import Cam3DTracker, track_bbox3d2result


@MODELS.register_module()
class JDT3D(Cam3DTracker):
    NUS_PEDESTRIAN_IDX = 6
    NUS_TRAFFIC_CONE_IDX = 8
    def __init__(
            self, 
            *args, 
            voxelize_cfg:dict, 
            voxelize_reduce:bool,
            feat_channels:int=512, # lidar backbone output
            hidden_channel:int=128,
            view_transform:Optional[dict]=None,
            fusion_layer:Optional[dict]=None,
            batch_clip:bool=True,
            heatmap_query_init:bool=False,
            heatmap_nms_kernel:int=3,
            **kwargs
        ):
        """Fusion3DTracker.
        Args:
            batch_clip (bool, optional): Whether to put frames from clip into a single
            batch. If out of memory errors, set to False.
        """
        super().__init__(*args, **kwargs)
        if self.num_classes == 10:
            self.nontracking_classes = True
        else:
            assert self.num_classes == 7, "Only support 7 classes for tracking"
            self.nontracking_classes = False
        self.batch_clip = batch_clip
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)
        self.voxelize_reduce = voxelize_reduce

        if fusion_layer is not None:
            self.fusion_layer = MODELS.build(fusion_layer)
            assert view_transform is not None, "view_transform should be provided when fusion_layer is not None"
            self.view_transform = MODELS.build(view_transform)
        if not self.train_backbone: # needs to be called after super init
            for param in self.pts_voxel_encoder.parameters():
                param.requires_grad = False
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
            for param in self.pts_neck.parameters():
                param.requires_grad = False
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False
            if self.with_fusion:
                for param in self.fusion_layer.parameters():
                    param.requires_grad = False

        # to go from input to the hidden channel
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            feat_channels, # 512
            hidden_channel, # 128
            kernel_size=3,
            padding=1,
            bias='auto',
        )
        self.heatmap_query_init = heatmap_query_init
        if self.heatmap_query_init:
            layers = []
            layers.append(
                ConvModule(
                    hidden_channel,
                    hidden_channel,
                    kernel_size=3,
                    padding=1,
                    bias='auto',
                    conv_cfg=dict(type='Conv2d'),
                    norm_cfg=dict(type='BN2d'),
                ))
            layers.append(
                build_conv_layer(
                    dict(type='Conv2d'),
                    hidden_channel,
                    self.num_classes,
                    kernel_size=3,
                    padding=1,
                    bias='auto',
                ))
            self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(self.num_classes, hidden_channel, 1)
            self.nms_kernel = heatmap_nms_kernel
            # x_size, y_size of feature map
            x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
            y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
            self.bev_pos = self.create_2D_grid(x_size, y_size)
            # append 0.5 to the end of the position embedding to represent the z dimension
            self.bev_pos = torch.cat([self.bev_pos, torch.zeros_like(self.bev_pos[:, :, :1]) + 0.5], dim=-1) # [1, x_size * y_size, 3]

    def init_weights(self) -> None:
        if self.with_img_backbone:
            self.img_backbone.init_weights()

    def create_2D_grid(self, x_size:int, y_size:int):
        """create meshgrid for 2D position embedding
        Each coordinate is in the range of [0, 1]
        The dimension of the output is [1, x_size * y_size, 2]        
        """
        meshgrid = [[0, 1, x_size], [0, 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5/x_size
        batch_y = batch_y + 0.5/y_size
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def loss(
            self,
            inputs:dict,
            data_samples):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                For each sample, the format is Dict(key: contents of the timestamps)
                Defaults to None. For each field, its shape is [T * NumCam * ContentLength]
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None. Number same as batch size.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, T, Num_Cam, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
            l2g_r_mat (list[Tensor]). Lidar to global transformation, shape [T, 3, 3]
            l2g_t (list[Tensor]). Lidar to global rotation
                points @ R_Mat + T
            timestamp (list). Timestamp of the frames
        Returns:
            dict: Losses of different branches.
        """
        points = inputs['points']
        num_frame = len(points)

        # extract the features of multi-frame images and points
        fused_feats = self.extract_clip_feats(inputs, data_samples)

        # Empty the runtime_tracker
        # Use PETR head to decode the bounding boxes on every frame
        outs = list()
        # returns only single set of track instances, corresponding to single batch
        if self.heatmap_query_init:
            next_frame_track_instances, dense_heatmap = self.generate_heatmap_query_instance(fused_feats[0])
        else:
            next_frame_track_instances = self.generate_empty_instance()
            dense_heatmap = None
        device = next_frame_track_instances.reference_points.device

        # Running over all the frames one by one
        self.runtime_tracker.empty()
        for frame_idx, data_samples_frame in enumerate(data_samples):
            img_metas_single_frame = [ds_batch.metainfo for ds_batch in data_samples_frame]
            ff_gt_bboxes_list = [ds_batch.gt_instances_3d.bboxes_3d for ds_batch in data_samples_frame]
            ff_gt_labels_list = [ds_batch.gt_instances_3d.labels_3d for ds_batch in data_samples_frame]
            ff_instance_inds = [ds_batch.gt_instances_3d.instance_inds for ds_batch in data_samples_frame]
            gt_forecasting_locs = data_samples_frame[0].gt_instances_3d.forecasting_locs
            gt_forecasting_masks = data_samples_frame[0].gt_instances_3d.forecasting_masks

            # PETR detection head
            track_instances = next_frame_track_instances
            out = self.pts_bbox_head(
                fused_feats[frame_idx], img_metas_single_frame, 
                track_instances.query_feats, track_instances.query_embeds, 
                track_instances.reference_points)
            
            # add heatmap to out dict for loss computation
            out['dense_heatmap'] = dense_heatmap

            # 1. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances

            # 2. Loss computation for the detection
            out['loss_dict'] = self.criterion.loss_single_frame(
                frame_idx, ff_gt_bboxes_list, ff_gt_labels_list,
                ff_instance_inds, out, None)

            # 3. Spatial-temporal reasoning
            track_instances = self.STReasoner(track_instances)

            if self.STReasoner.history_reasoning:
                out['loss_dict'] = self.criterion.loss_mem_bank(
                    frame_idx,
                    out['loss_dict'],
                    ff_gt_bboxes_list,
                    ff_gt_labels_list,
                    ff_instance_inds,
                    track_instances)

            if self.STReasoner.future_reasoning:
                active_mask = (track_instances.obj_idxes >= 0)
                out['loss_dict'] = self.forward_loss_prediction(
                    frame_idx,
                    out['loss_dict'],
                    track_instances[active_mask],
                    gt_forecasting_locs,
                    gt_forecasting_masks,
                    ff_instance_inds)

            # 4. Prepare for next frame if not on the last frame
            if frame_idx < num_frame - 1:
                track_instances = self.frame_summarization(track_instances, tracking=False)
                # active_mask = self.runtime_tracker.get_active_mask(track_instances, training=True)
                active_mask = (track_instances.scores > self.runtime_tracker.threshold)
                track_instances.track_query_mask[active_mask] = True
                active_track_instances = track_instances[active_mask]
                if self.motion_prediction:
                    # assume batch size is 1
                    time_delta = data_samples[frame_idx+1][0].metainfo['timestamp'] - data_samples[frame_idx][0].metainfo['timestamp']
                    active_track_instances = self.update_reference_points(
                        active_track_instances,
                        time_delta,
                        use_prediction=self.motion_prediction_ref_update,
                        tracking=False)
                if self.if_update_ego:
                    active_track_instances = self.update_ego(
                        active_track_instances, 
                        data_samples[frame_idx][0].metainfo['lidar2global'].to(device), 
                        data_samples[frame_idx + 1][0].metainfo['lidar2global'].to(device),
                    )
                active_track_instances = self.STReasoner.sync_pos_embedding(active_track_instances, self.query_embedding)
            
                if self.heatmap_query_init:
                    empty_track_instances, dense_heatmap = self.generate_heatmap_query_instance(fused_feats[frame_idx+1])
                else:
                    empty_track_instances = self.generate_empty_instance()
                    dense_heatmap = None
                next_frame_track_instances = Instances.cat([empty_track_instances, active_track_instances])

            self.runtime_tracker.frame_index += 1
            outs.append(out)
        losses = self.criterion(outs)
        self.runtime_tracker.empty()
        return losses

    def predict(self, inputs:dict, data_samples):
        points = inputs['points']
        num_frame = len(points)
        batch_size = len(points[0])
        assert num_frame == 1, "Only support single frame prediction"
        assert batch_size == 1, "Only support single bs prediction"

        # extract the features of multi-frame images and points
        fused_feats = self.extract_clip_feats(inputs, data_samples)
        
        # new sequence
        timestamp = data_samples[0][0].metainfo['timestamp']
        if self.runtime_tracker.timestamp is None or abs(timestamp - self.runtime_tracker.timestamp) > 10:
            self.runtime_tracker.timestamp = timestamp
            self.runtime_tracker.current_seq += 1
            self.runtime_tracker.track_instances = None
            self.runtime_tracker.current_id = 0
            self.runtime_tracker.l2g = None
            self.runtime_tracker.time_delta = 0
            self.runtime_tracker.frame_index = 0
        self.runtime_tracker.time_delta = timestamp - self.runtime_tracker.timestamp
        self.runtime_tracker.timestamp = timestamp

        # processing the queries from t-1
        prev_active_track_instances = self.runtime_tracker.track_instances
        for frame_idx in range(num_frame): # TODO remove this for loop, assume num_frame = 1 for prediction
            img_metas_single_frame = [ds[frame_idx].metainfo for ds in data_samples]
        
            # 1. Update the information of previous active tracks
            if self.heatmap_query_init:
                empty_track_instances, _ = self.generate_heatmap_query_instance(fused_feats[frame_idx])
            else:
                empty_track_instances = self.generate_empty_instance()

            if prev_active_track_instances is None:
                track_instances = empty_track_instances
            else:
                device = prev_active_track_instances.reference_points.device
                if self.motion_prediction:
                    time_delta = self.runtime_tracker.time_delta
                    prev_active_track_instances = self.update_reference_points(
                        prev_active_track_instances, time_delta, 
                        use_prediction=self.motion_prediction_ref_update, tracking=True)
                if self.if_update_ego:
                    prev_active_track_instances = self.update_ego(
                        prev_active_track_instances, self.runtime_tracker.l2g.to(device), 
                        img_metas_single_frame[0]['lidar2global'].to(device))
                prev_active_track_instances = self.STReasoner.sync_pos_embedding(prev_active_track_instances, self.query_embedding)
                track_instances = Instances.cat([empty_track_instances, prev_active_track_instances])

            self.runtime_tracker.l2g = img_metas_single_frame[0]['lidar2global']
            self.runtime_tracker.timestamp = img_metas_single_frame[0]['timestamp']

            # 2. PETR detection head
            out = self.pts_bbox_head(
                fused_feats[frame_idx], img_metas_single_frame, track_instances.query_feats,
                track_instances.query_embeds, track_instances.reference_points)
            # 3. Record the information into the track instances cache
            track_instances = self.load_detection_output_into_cache(track_instances, out)
            out['track_instances'] = track_instances

            # 4. Spatial-temporal Reasoning
            self.STReasoner(track_instances)
            track_instances = self.frame_summarization(track_instances, tracking=True)
            out['all_cls_scores'][-1][0, :] = track_instances.logits
            out['all_bbox_preds'][-1][0, :] = track_instances.bboxes

            if self.STReasoner.future_reasoning:
                # motion forecasting has the shape of [num_query, T, 2]
                out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
            else:
                out['all_motion_forecasting'] = None

            # 5. Track class filtering: before decoding bboxes, only leave the objects under tracking categories
            if self.tracking:
                max_cat = torch.argmax(out['all_cls_scores'][-1, 0, :].sigmoid(), dim=-1)
                related_cat_mask = (max_cat < 7) # we set the first 7 classes as the tracking classes of nuscenes
                track_instances = track_instances[related_cat_mask]
                out['all_cls_scores'] = out['all_cls_scores'][:, :, related_cat_mask, :]
                out['all_bbox_preds'] = out['all_bbox_preds'][:, :, related_cat_mask, :]
                if out['all_motion_forecasting'] is not None:
                    out['all_motion_forecasting'] = out['all_motion_forecasting'][related_cat_mask, ...]

                # 6. assign ids
                active_mask = (track_instances.scores > self.runtime_tracker.threshold)
                for i in range(len(track_instances)):
                    if track_instances.obj_idxes[i] < 0:
                        track_instances.obj_idxes[i] = self.runtime_tracker.current_id 
                        self.runtime_tracker.current_id += 1
                        if active_mask[i]:
                            track_instances.track_query_mask[i] = True
                out['track_instances'] = track_instances

                # 7. Prepare for the next frame and output
                score_mask = (track_instances.scores > self.runtime_tracker.output_threshold)
                out['all_masks'] = score_mask.clone()
                self.runtime_tracker.update_active_tracks(track_instances, active_mask)

            bbox_list = self.pts_bbox_head.get_bboxes(out, img_metas_single_frame, tracking=True)
            # self.runtime_tracker.update_active_tracks(active_track_instances)

            # each time, only run one frame
            self.runtime_tracker.frame_index += 1
            break

        bbox_results = [
            track_bbox3d2result(bboxes, scores, labels, obj_idxes, track_scores, forecasting)
            for bboxes, scores, labels, obj_idxes, track_scores, forecasting in bbox_list
        ]
        if self.tracking:
            bbox_results[0]['track_ids'] = [f'{self.runtime_tracker.current_seq}-{i}' for i in bbox_results[0]['track_ids'].long().cpu().numpy().tolist()]
        results_list_3d = []
        for results in bbox_results:
            instance = InstanceData()
            # instance = InstanceData(metainfo=results)
            instance.bboxes_3d = results['bboxes_3d']
            instance.labels_3d = results['labels_3d']
            instance.scores_3d = results['scores_3d']
            instance.track_scores = results['track_scores']
            instance.track_ids = results['track_ids']
            instance.forecasting = results['forecasting']
            results_list_3d.append(instance)
        detsamples = self.add_pred_to_datasample(
            data_samples[0], data_instances_3d=results_list_3d
        )
        return detsamples


    def extract_clip_feats(self, inputs, data_samples):
        outputs = list()
        pts_clip = inputs['points']
        num_frames = len(pts_clip)
        batch_size = len(pts_clip[0])
        img_clip = inputs.get('imgs', [None]*num_frames)
        if self.batch_clip:
            # put batched frames from clip into a single superbatch
            # single frame image, N * NumCam * C * H * W
            if img_clip[0] is not None:
                imgs_stacked = torch.cat([frame for frame in img_clip], dim=0)
            else:
                imgs_stacked = None
            pts_stacked = []
            input_metas = []
            for pts_i, data_sample_i in zip(pts_clip, data_samples):
                pts_stacked.extend(pts_i)
                input_metas.extend([ds_batch.metainfo for ds_batch in data_sample_i])
            # extract features from superbatch
            input_dict = dict(imgs=imgs_stacked, points=pts_stacked)
            fused_feats = self.extract_feat(input_dict, input_metas)
            # extract output from superbatch back to clip
            for frame_idx in range(num_frames):
                outputs.append(fused_feats[frame_idx * batch_size:(frame_idx + 1) * batch_size])
        else:
            # process each frame in clip separately
            for frame_idx, (img_frame, pts_frame) in enumerate(zip(img_clip, pts_clip)):
                input_dict = dict(imgs=img_frame, points=pts_frame)
                # iterate over each item in batch for given frame_idx
                input_metas = [item[frame_idx].metainfo for item in data_samples]
                fused_feats = self.extract_feat(input_dict, input_metas)
                outputs.append(fused_feats)
        return outputs

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                num_cams = len(meta['cam2img'])
                img_aug_matrix.append(
                    meta.get('img_aug_matrix', 
                             np.vstack([np.eye(4)]*num_cams).reshape(num_cams, 4, 4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.with_fusion:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)[0]
        x = self.shared_conv(x)
        return x

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x
    
    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def generate_heatmap_query_instance(self, fused_feat:torch.Tensor):
        empty_track_instances = self.generate_empty_instance()

        batch_size = fused_feat.shape[0]
        fused_feat_flatten = fused_feat.view(batch_size, fused_feat.shape[1], -1) # [B, C, H*W]
        dense_heatmap = self.heatmap_head(fused_feat)
        # technically sigmoid here is optional since we only use the max value.
        # sigmoid is applied again in the loss computation
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner

        assert 'dataset' in self.test_cfg, 'dataset should be specified in test_cfg'
        if self.test_cfg['dataset'] == 'nuScenes': # for Pedestrian & Traffic_cone in nuScenes
            # set kernel for pedestrian to 1
            local_max[:, self.NUS_PEDESTRIAN_IDX, ] = F.max_pool2d(
                heatmap[:, self.NUS_PEDESTRIAN_IDX], kernel_size=1, stride=1, padding=0)
            # set kernel for traffic_cone to 1
            if self.nontracking_classes:
                local_max[:, self.NUS_TRAFFIC_CONE_IDX, ] = F.max_pool2d(
                    heatmap[:, self.NUS_TRAFFIC_CONE_IDX], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            raise NotImplementedError
            # local_max[:, 1, ] = F.max_pool2d(
            #     heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            # local_max[:, 2, ] = F.max_pool2d(
            #     heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        else:
            raise NotImplementedError
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

        # top num_proposals among all classes
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_query]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = fused_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fused_feat_flatten.shape[1], -1),
            dim=-1,
        )

        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(query_feat.device)
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1,
        )

        """Detection queries"""
        # remove batch dim
        empty_track_instances.reference_points = query_pos[0].clone()
        # convert query_feat from [C, num_query] to [num_query, C]
        empty_track_instances.query_feats = query_feat[0].clone().permute(1, 0)
        empty_track_instances.query_embeds = self.query_embedding(pos2posemb3d(query_pos[0]))
        """Cache for current frame information, loading temporary data for spatial-temporal reasoning"""
        empty_track_instances.cache_reference_points = empty_track_instances.reference_points.clone()
        empty_track_instances.cache_query_embeds = empty_track_instances.query_embeds.clone()
        empty_track_instances.cache_query_feats = empty_track_instances.query_feats.clone()

        return empty_track_instances, dense_heatmap

    @property
    def with_fusion(self):
        """bool: Whether the detector has a fusion layer.
        bug fix on mvxtwostage
        """
        return hasattr(self,
                       'fusion_layer') and self.fusion_layer is not None
