from typing import List

import numpy as np
from mmengine.dataset import BaseDataset
from mmdet3d.datasets import CBGSDataset

from mmdet3d.registry import DATASETS

@DATASETS.register_module()
class CBGSDataset2(CBGSDataset):
    def _get_sample_indices(self, dataset: BaseDataset) -> List[int]:
        """Load sample indices according to ann_file.

        Args:
            dataset (:obj:`BaseDataset`): The dataset.

        Returns:
            List[dict]: List of indices after class sampling.
        """
        classes = self.metainfo['classes']
        cat2id = {name: i for i, name in enumerate(classes)}
        class_sample_idxs = {cat_id: [] for cat_id in cat2id.values()}
        for idx in range(len(dataset)):
            sample_cat_ids = dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                if cat_id != -1:
                    # Filter categories that do not need to be cared.
                    # -1 indicates dontcare in MMDet3D.
                    class_sample_idxs[cat_id].append(idx)
        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: max(1, len(v)) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices