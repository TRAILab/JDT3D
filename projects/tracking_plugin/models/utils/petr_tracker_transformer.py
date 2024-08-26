# ------------------------------------------------------------------------
# Copyright (c) 2022 toyota research institute
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

from mmdet3d.registry import MODELS

from projects.PETR.petr.petr_transformer import PETRTransformer


@MODELS.register_module()
class PETRTrackingTransformer(PETRTransformer):
    """Implements the DETR transformer.
    Adding a field of target to denote the query features.
    If target is None, the behavior is identical to PETRTransformer.
    """
    
    def forward(self, target, x, mask, query_embed, pos_embed, reg_branch=None):
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        target = target.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]

        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
            reg_branch=reg_branch,
            )
        out_dec = out_dec.transpose(1, 2)
        memory = memory.reshape(n, h, w, bs, c).permute(3, 0, 4, 1, 2)
        return  out_dec, memory
