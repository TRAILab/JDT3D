from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet.models.layers.transformer import inverse_sigmoid

from .petr_tracking_head import PETRCamTrackingHead


@MODELS.register_module()
class BEVFusionTrackingHead(PETRCamTrackingHead):
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
        x = mlvl_feats
        batch_size = x.size(0)
        bev_h, bev_w = x.size(2), x.size(3)
        x = self.input_proj(x)
        masks = x.new_zeros((batch_size, bev_h, bev_w)).to(torch.bool)
        x = x.view(batch_size, 1, *x.shape[-3:]) # add a dimension to simulate num_cams for transformer compatibility

        # Key difference with PETR tracking head is the position embedding.
        # Because we work with BEV features, we don't need the extra 2D to 3D stuff in PETR.
        pos_embed = self.positional_encoding(masks)
        pos_embed.unsqueeze_(1) # add a dimension to simulate num_cams for transformer compatibility

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