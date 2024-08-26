# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
from typing import List, Union

import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import ForwardResults, OptSampleList
from mmengine.structures import InstanceData

from projects.PETR.petr.petr import PETR
from projects.PETR.petr.petr_head import pos2posemb3d
from projects.tracking_plugin.core.instances import Instances

from .runtime_tracker import RunTimeTracker
from .spatial_temporal_reason import SpatialTemporalReasoner


@MODELS.register_module()
class Cam3DTracker(PETR):
    def __init__(self,
                 num_classes=10,
                 num_query=100,
                 tracking=True,
                 train_backbone=True,
                 if_update_ego=True,
                 motion_prediction=True,
                 motion_prediction_ref_update=True,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                 spatial_temporal_reason=None,
                 runtime_tracker=None,
                 loss=None,
                 **kwargs):
        super(Cam3DTracker, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_query = num_query
        self.embed_dims = 256
        self.tracking = tracking
        self.train_backbone = train_backbone
        self.if_update_ego = if_update_ego
        self.motion_prediction = motion_prediction
        self.motion_prediction_ref_update=motion_prediction_ref_update
        self.pc_range = pc_range
        self.position_range = position_range
        self.criterion = MODELS.build(loss)

        # spatial-temporal reasoning
        self.STReasoner = SpatialTemporalReasoner(**spatial_temporal_reason)
        self.hist_len = self.STReasoner.hist_len
        self.fut_len = self.STReasoner.fut_len

        self.init_params_and_layers()

        # Inference time tracker
        self.runtime_tracker = RunTimeTracker(**runtime_tracker)
    
    def forward(
            self, 
            inputs: Union[dict, List[dict]], 
            data_samples: OptSampleList = None, 
            mode: str = 'loss', 
            **kwargs) -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict mode')

    def generate_empty_instance(self):
        """Generate empty instance slots at the beginning of tracking"""
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device

        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        track_instances.reference_points = reference_points.clone()
        track_instances.query_embeds = query_embeds.clone()
        if self.tracking:
            track_instances.query_feats = self.query_feat_embedding.weight.clone()
        else:
            track_instances.query_feats = torch.zeros_like(query_embeds)

        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)

        """Current frame information"""
        # classification scores
        track_instances.logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # bounding boxes
        track_instances.bboxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        
        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.cache_bboxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_reference_points = reference_points.clone()
        track_instances.cache_query_embeds = query_embeds.clone()
        if self.tracking:
            track_instances.cache_query_feats = self.query_feat_embedding.weight.clone()
        else:
            track_instances.cache_query_feats = torch.zeros_like(query_embeds)
        track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)

        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)

        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)
        return track_instances

    def update_ego(self, track_instances, l2g0, l2g1):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        """
        track_instances = self.STReasoner.update_ego(track_instances, l2g0, l2g1)
        return track_instances
    
    def update_reference_points(self, track_instances, time_delta=None, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction/velocities
        """
        track_instances = self.STReasoner.update_reference_points(
            track_instances, time_delta, use_prediction, tracking)
        return track_instances
    
    def load_detection_output_into_cache(self, track_instances: Instances, out):
        """ Load output of the detection head into the track_instances cache (inplace)
        """
        query_feats = out.pop('query_feats')
        query_reference_points = out.pop('reference_points')
        with torch.no_grad():
            track_scores = out['all_cls_scores'][-1, 0, :].sigmoid().max(dim=-1).values
        track_instances.cache_scores = track_scores.clone()
        track_instances.cache_logits = out['all_cls_scores'][-1, 0].clone()
        track_instances.cache_query_feats = query_feats[0].clone()
        track_instances.cache_reference_points = query_reference_points[0].clone()
        track_instances.cache_bboxes = out['all_bbox_preds'][-1, 0].clone()
        track_instances.cache_query_embeds = self.query_embedding(pos2posemb3d(track_instances.cache_reference_points))
        return track_instances
    
    def forward_loss_prediction(self, 
                                frame_idx,
                                loss_dict,
                                active_track_instances,
                                gt_trajs,
                                gt_traj_masks,
                                instance_inds,):
        active_gt_trajs, active_gt_traj_masks = list(), list()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(
            instance_inds[0].detach().cpu().numpy().tolist())}

        active_gt_trajs = torch.ones_like(active_track_instances.motion_predictions)
        active_gt_trajs[..., -1] = 0.0
        active_gt_traj_masks = torch.zeros_like(active_gt_trajs)[..., 0]
        for track_idx, id in enumerate(active_track_instances.obj_idxes):
            cpu_id = id.cpu().numpy().tolist()
            if cpu_id not in obj_idx_to_gt_idx.keys():
                continue
            index = obj_idx_to_gt_idx[cpu_id]
            traj = gt_trajs[index:index+1, :self.fut_len + 1, :]
            gt_motion = traj[:, torch.arange(1, self.fut_len + 1)] - traj[:, torch.arange(0, self.fut_len)]
            active_gt_trajs[track_idx: track_idx + 1] = gt_motion
            active_gt_traj_masks[track_idx: track_idx + 1] = \
                gt_traj_masks[index: index+1, 1: self.fut_len + 1] * gt_traj_masks[index: index+1, : self.fut_len]
        
        loss_dict = self.criterion.loss_prediction(frame_idx,
                                                   loss_dict,
                                                   active_gt_trajs[..., :2],
                                                   active_gt_traj_masks,
                                                   active_track_instances.cache_motion_predictions[..., :2])
        return loss_dict
    
    def frame_summarization(self, track_instances, tracking=False):
        """ Load the results after spatial-temporal reasoning into track instances
        """
        # inference mode
        if tracking:
            active_mask = (track_instances.cache_scores >= self.runtime_tracker.record_threshold)
        # training mode
        else:
            track_instances.bboxes = track_instances.cache_bboxes.clone()
            track_instances.logits = track_instances.cache_logits.clone()
            track_instances.scores = track_instances.cache_scores.clone()
            active_mask = (track_instances.cache_scores >= self.runtime_tracker.record_threshold)

        track_instances.query_feats[active_mask] = track_instances.cache_query_feats[active_mask]
        track_instances.query_embeds[active_mask] = track_instances.cache_query_embeds[active_mask]
        track_instances.logits[active_mask] = track_instances.cache_logits[active_mask]
        track_instances.scores[active_mask] = track_instances.cache_scores[active_mask]
        track_instances.motion_predictions[active_mask] = track_instances.cache_motion_predictions[active_mask]
        track_instances.bboxes[active_mask] = track_instances.cache_bboxes[active_mask]
        track_instances.reference_points[active_mask] = track_instances.cache_reference_points[active_mask]

        # TODO: generate future bounding boxes, reference points, scores
        if self.STReasoner.future_reasoning:
            motion_predictions = track_instances.motion_predictions[active_mask]
            track_instances.fut_xyz[active_mask] = track_instances.reference_points[active_mask].clone()[:, None, :].repeat(1, self.fut_len, 1)
            track_instances.fut_bboxes[active_mask] = track_instances.bboxes[active_mask].clone()[:, None, :].repeat(1, self.fut_len, 1)
            motion_add = torch.cumsum(motion_predictions.clone().detach(), dim=1)
            motion_add_normalized = motion_add.clone()
            motion_add_normalized[..., 0] /= (self.pc_range[3] - self.pc_range[0])
            motion_add_normalized[..., 1] /= (self.pc_range[4] - self.pc_range[1])
            track_instances.fut_xyz[active_mask, :, 0] += motion_add_normalized[..., 0]
            track_instances.fut_xyz[active_mask, :, 1] += motion_add_normalized[..., 1]
            track_instances.fut_bboxes[active_mask, :, 0] += motion_add[..., 0]
            track_instances.fut_bboxes[active_mask, :, 1] += motion_add[..., 1]
        return track_instances
   
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
        img = inputs['imgs']
        num_frame = len(img)
        batch_size = img[0].shape[0]
        assert batch_size == 1, "Currently only support batch size 1"

        # Image features, one clip at a time for checkpoint usages
        img_feats = self.extract_clip_imgs_feats(img=img)

        # Empty the runtime_tracker
        # Use PETR head to decode the bounding boxes on every frame
        outs = list()
        # returns only single set of track instances, corresponding to single batch
        next_frame_track_instances = self.generate_empty_instance()
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
                img_feats[frame_idx], img_metas_single_frame, 
                track_instances.query_feats, track_instances.query_embeds, 
                track_instances.reference_points)

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

            # 4. Prepare for next frame
            track_instances = self.frame_summarization(track_instances, tracking=False)
            active_mask = self.runtime_tracker.get_active_mask(track_instances, training=True)
            track_instances.track_query_mask[active_mask] = True
            active_track_instances = track_instances[active_mask]
            if self.motion_prediction and frame_idx < num_frame - 1:
                # assume batch size is 1
                time_delta = data_samples[frame_idx+1][0].metainfo['timestamp'] - data_samples[frame_idx][0].metainfo['timestamp']
                active_track_instances = self.update_reference_points(
                    active_track_instances,
                    time_delta,
                    use_prediction=self.motion_prediction_ref_update,
                    tracking=False)
            if self.if_update_ego and frame_idx < num_frame - 1:
                active_track_instances = self.update_ego(
                    active_track_instances, 
                    data_samples[frame_idx][0].metainfo['lidar2global'].to(device), 
                    data_samples[frame_idx + 1][0].metainfo['lidar2global'].to(device),
                )
            if frame_idx < num_frame - 1:
                active_track_instances = self.STReasoner.sync_pos_embedding(active_track_instances, self.query_embedding)
            
            empty_track_instances = self.generate_empty_instance()
            next_frame_track_instances = Instances.cat([empty_track_instances, active_track_instances])
            self.runtime_tracker.frame_index += 1
            outs.append(out)
        losses = self.criterion(outs)
        self.runtime_tracker.empty()
        return losses

    def predict(self, inputs:dict, data_samples):
        imgs = inputs['imgs'] # bs, num frames, num cameras (6), C, H, W
        batch_size = len(imgs)
        assert batch_size == 1, "Only support single bs prediction"
        num_frame = imgs[0].shape[0]
        assert num_frame == 1, "Only support single frame prediction"

        # backbone images
        img_feats = self.extract_clip_imgs_feats(img=imgs)
        
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
            if prev_active_track_instances is None:
                track_instances = self.generate_empty_instance()
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
                track_instances = Instances.cat([self.generate_empty_instance(), prev_active_track_instances])

            self.runtime_tracker.l2g = img_metas_single_frame[0]['lidar2global']
            self.runtime_tracker.timestamp = img_metas_single_frame[0]['timestamp']

            # 2. PETR detection head
            out = self.pts_bbox_head(
                img_feats[frame_idx], img_metas_single_frame, track_instances.query_feats,
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

        results_list_3d = [InstanceData(metainfo=results) for results in bbox_results]
        detsamples = self.add_pred_to_datasample(
            data_samples[0], data_instances_3d=results_list_3d
        )
        return detsamples

    def extract_clip_imgs_feats(self, img):
        """Extract the features of multi-frame images
        Args:
            img_metas (list[dict], optional): Meta information of each sample.
                For each sample, the format is Dict(key: contents of the timestamps)
                Defaults to None. For each field, its shape is [T * NumCam * ContentLength]
            img (torch.Tensor optional): Images of each sample with shape
                (N, T, Num_Cam, C, H, W). Defaults to None.
        Return:
           img_feats (list[torch.Tensor]): List of features on every frame.
        """
        num_frame = len(img)
        batch_size, num_cam = img[0].shape[0], img[0].shape[1]

        # backbone images
        # get all the images and let backbone infer for once
        dummy_img_metas = [dict() for _ in range(batch_size)] # metas not actually required to extract features

        # all imgs B * (T * NumCam) * C * H * W, T is frame num
        all_imgs = torch.cat([frame for frame in img], dim=1)
        # img_feats List[Tensor of batch 0, ...], each tensor BS * (T * NumCam) * C * H * W
        all_img_feats = self.extract_feat(img=all_imgs, img_metas=dummy_img_metas)

        # per frame feature maps
        img_feats = list()
        for i in range(num_frame):
            single_frame_feats = [lvl_feats[:, num_cam * i: num_cam * (i + 1), :, :, :] for lvl_feats in all_img_feats]
            img_feats.append(single_frame_feats)
        return img_feats
    
    def init_params_and_layers(self):
        """Generate the instances for tracking, especially the object queries
        """
        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

        # embedding initialization for tracking
        if self.tracking:
            self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)
            nn.init.zeros_(self.query_feat_embedding.weight)
        
        # freeze backbone
        if not self.train_backbone and self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        return


def track_bbox3d2result(bboxes, scores, labels, obj_idxes, track_scores, forecasting=None, attrs=None):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        forecasting (torch.Tensor): Motion forecasting with shape of (n, T, 2)
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.
    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - forecasting (torh.Tensor, optional): Motion forecasting
    """
    result_dict = dict(
        bboxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu(),
        track_scores=track_scores.cpu(),
        track_ids=obj_idxes.cpu(),
    )

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()
    
    if forecasting is not None:
        result_dict['forecasting'] = forecasting.cpu()[..., :2]

    return result_dict