import math
from typing import Optional

import numpy as np
from mmdet.datasets.samplers import TrackImgSampler
from mmdet3d.registry import DATA_SAMPLERS
from mmengine.dist import get_dist_info, sync_random_seed

from projects.tracking_plugin.datasets.nuscenes_tracking_dataset import (
    NuScenesTrackingDataset,
)


@DATA_SAMPLERS.register_module()
class TrackSampler3D(TrackImgSampler):
    def __init__(
        self,
        dataset: NuScenesTrackingDataset,
        seed: Optional[int] = None,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        if seed is None:
            self.seed = sync_random_seed()
        else:
            self.seed = seed

        self.dataset = dataset

        # list of length world_size, each element is a list of tuples (video_ind, frame_ind) for that gpu
        self.indices = []

        # Hard code here to handle different dataset wrapper
        assert isinstance(
            self.dataset, NuScenesTrackingDataset
        ), "TrackImgSampler is only supported in NuScenesTrackingDataset but"
        f"got {type(self.dataset)} "
        # TODO support CBGS wrapper
        self.test_mode = self.dataset.test_mode
        scene_tokens = self.dataset.get_all_scene_tokens()
        if self.test_mode:
            # in test mode, the images belong to the same video must be put
            # on the same device.
            if len(scene_tokens) < self.world_size:
                raise ValueError(
                    f"only {len(scene_tokens)} videos loaded,"
                    f"but {self.world_size} gpus were given."
                )
            scene_token_splits = np.array_split(scene_tokens, self.world_size)
            for scene_token_subset in scene_token_splits:
                indices_chunk = []
                for scene_token in scene_token_subset:
                    indices_chunk.extend(self.dataset.indices_per_scene[scene_token])
                self.indices.append(indices_chunk)
        else:
            for scene_token in scene_tokens:
                self.indices.extend(self.dataset.indices_per_scene[scene_token])

        if self.test_mode:
            self.num_samples = len(self.indices[self.rank])
            self.total_size = sum([len(index_list) for index_list in self.indices])
        else:
            self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.world_size))
            self.total_size = self.num_samples * self.world_size
