# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from MUTR3D (https://github.com/a1600012888/MUTR3D)
# Copyright (c) 2022 Tianyuan Zhang
# ------------------------------------------------------------------------
from mmdet3d.datasets.transforms.loading import LoadAnnotations3D
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class TrackLoadAnnotations3D(LoadAnnotations3D):

    def __init__(self,
                 with_forecasting=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.with_forecasting = with_forecasting
    
    def _load_track_ids(self, results):
        results['instance_inds'] = results['ann_info']['instance_inds']
        return results
    
    def _load_forecasting(self, results):
        """Private function to load forecasting annotations
        """
        results['gt_forecasting_locs'] = results['ann_info']['forecasting_locs']
        results['gt_forecasting_masks'] = results['ann_info']['forecasting_masks']
        results['gt_forecasting_types'] = results['ann_info']['forecasting_types']
        return results

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        results = self._load_track_ids(results)
        if self.with_forecasting:
            results = self._load_forecasting(results)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = super().__repr__() + f'{indent_str}with_forecasting={self.with_forecasting}, '

        return repr_str