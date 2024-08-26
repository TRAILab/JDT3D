# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute
# ------------------------------------------------------------------------
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init


@MODELS.register_module()
class TemporalTransformer(BaseModule):
    """Implement a DETR transformer.
    Adapting the input and output to the motion reasoning purpose.
    """
    def __init__(self, encoder=None, decoder=None, init_cfg=None, cross=False):
        super(TemporalTransformer, self).__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        else:
            self.encoder = None
        self.decoder = MODELS.build(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.cross = cross
        
    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        return
    
    def forward(self, target, x, query_embed, pos_embed, query_key_padding_mask=None, key_padding_mask=None):
        """ The general transformer interface for temporal/spatial cross attention
        Args:
            target: query feature [num_query, len, dim]
            x: key/value features [num_query, len, dim]
            query_embed: query positional embedding [num_query, len, dim]
            pos_embed: key positional embedding [num_query, len, dim]
        """
        # suit the shape for transformer
        bs = 1
        memory = x.transpose(0, 1)
        pos_embed = pos_embed.transpose(0, 1)
        query_embed = query_embed.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        target = target.transpose(0, 1)  # [num_query, dim] -> [num_query, bs, dim]
        
        # if query_key_padding_mask is not None:
        #     query_key_padding_mask = query_key_padding_mask.transpose(0, 1)
        
        # if key_padding_mask is not None:
        #     key_padding_mask = key_padding_mask.transpose(0, 1)
        
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            query_key_padding_mask=query_key_padding_mask,
            key_padding_mask=key_padding_mask)[-1]
        out_dec = out_dec.transpose(0, 1)
        return out_dec