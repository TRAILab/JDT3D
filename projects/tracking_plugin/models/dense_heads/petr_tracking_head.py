# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet.models.layers.transformer import inverse_sigmoid

from projects.PETR.petr.petr_head import PETRHead


@MODELS.register_module()
class PETRCamTrackingHead(PETRHead):
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRCamTrackingHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(PETRHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    
    def forward(
            self, 
            mlvl_feats, 
            img_metas:List[Dict[str, Any]], 
            query_targets, 
            query_embeds, 
            reference_points, 
            proj_feature=None, 
            proj_pos=None):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            query_targets ([Tensor]): the query feature vector, 
                replacing the all zeros in PETR
            query_embeds ([Tensor]): the query embeddings for decoder, 
                same as the original PETR
            reference_points ([Tensor]): reference points of the queries
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        x = mlvl_feats[0]
        batch_size, num_cams = x.size(0), x.size(1)
        input_img_h, input_img_w = img_metas[0]['pad_shape']
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w = img_metas[img_id]['batch_input_shape']
            masks[img_id, :num_cams, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(
                mlvl_feats, img_metas, masks)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
            sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(
                x.size())
            pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(
                    x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        # outs_dec [num_dec, bs, num_query, embed_dim]
        if proj_feature is not None: # depreciated, never called in Tracker level since proj_feature not passed
            outs_dec, _ = self.transformer(query_targets, x, masks, query_embeds, pos_embed, 
                                           self.reg_branches, proj_feature, proj_pos)
        else:
            outs_dec, _ = self.transformer(query_targets, x, masks, query_embeds, pos_embed, 
                                           self.reg_branches)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl]).to(
                torch.float32)
            tmp = self.reg_branches[lvl](outs_dec[lvl]).to(torch.float32)

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # last level
            if lvl == outs_dec.shape[0] - 1:
                last_reference_points = torch.cat((tmp[..., 0:2], tmp[..., 4:5]), dim=-1)

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) +
            self.pc_range[0])
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) +
            self.pc_range[1])
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) +
            self.pc_range[2])

        # record the query features for the next frame
        # pick the results from the last decoder
        last_query_feats = outs_dec[-1]

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
            'query_feats': last_query_feats,
            'reference_points': last_reference_points
        }
        return outs

    def get_bboxes(self, preds_dicts, img_metas, tracking=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            if tracking:
                track_scores = preds['track_scores']
                obj_idxes = preds['obj_idxes']
            else:
                obj_idxes = None
                track_scores = None
            
            forecasting = preds['forecasting']
            ret_list.append([bboxes, scores, labels, obj_idxes, track_scores, forecasting])
        return ret_list