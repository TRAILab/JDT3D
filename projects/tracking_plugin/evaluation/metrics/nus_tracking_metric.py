import logging
from os import path as osp
from typing import Dict, List, Optional, Sequence

import mmengine
import numpy as np
import pyquaternion
import torch
from mmdet3d.evaluation.metrics import NuScenesMetric
from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes
from mmengine.dist import broadcast_object_list, is_main_process
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import print_log
from nuscenes.eval.common.config import config_factory as track_configs

from projects.tracking_plugin.datasets import NuScenesForecastingBox
from projects.tracking_plugin.distr.custom_collect import custom_collect_results


@METRICS.register_module()
class NuScenesTrackingMetric(NuScenesMetric):
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]

    TRACKING_CLASSES = [
        "car",
        "truck",
        "bus",
        "trailer",
        "motorcycle",
        "bicycle",
        "pedestrian",
    ]

    KEYS = [
        "amota",
        "amotp",
        "recall",
        "motar",
        "gt",
        "mota",
        "motp",
        "mt",
        "ml",
        "faf",
        "tp",
        "fp",
        "fn",
        "ids",
        "frag",
        "tid",
        "lgd",
    ]

    def __init__(
        self, *args, tracking_eval_version="tracking_nips_2019", **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eval_tracking_configs = track_configs(tracking_eval_version)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample["pred_instances_3d"]
            pred_2d = data_sample["pred_instances"]
            for attr_name in pred_3d:
                # check if the attribute is a tensor
                if isinstance(pred_3d[attr_name], torch.Tensor):
                    pred_3d[attr_name] = pred_3d[attr_name].to("cpu")
            result["pred_instances_3d"] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to("cpu")
            result["pred_instances"] = pred_2d
            sample_idx = data_sample["sample_idx"]
            result["sample_idx"] = sample_idx
            self.results.append(result)

    def _format_lidar_bbox(
        self,
        results: List[dict],
        sample_idx_list: List[int],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None,
    ) -> str:
        nusc_annos = {}

        print("Start to convert detection format...")
        for i, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes, attrs = output_to_nusc_box(det)
            sample_idx = sample_idx_list[i]
            sample_token = self.data_infos[sample_idx]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_idx], boxes, classes, self.eval_detection_configs
            )
            for i, box in enumerate(boxes):
                name = classes[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = self.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    attribute_name=attr,
                )
                if box.forecasting is not None:
                    nusc_anno["forecasting"] = box.forecasting.tolist()
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }
        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc_tracking.json")
        print(f"Results writes to {res_path}")
        mmengine.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self,
        result_path: str,
        classes: Optional[List[str]] = None,
        result_name: str = "pred_instances_3d",
    ) -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """
        from nuscenes.eval.tracking.evaluate import TrackingEval

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            "v1.0-mini": "mini_val",
            # 'v1.0-mini': 'mini_train',
            "v1.0-trainval": "val",
            # 'v1.0-trainval': 'train',
        }
        detail = dict()
        try:
            nusc_eval = TrackingEval(
                config=self.eval_tracking_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            nusc_eval.main(render_curves=False)

            # record metrics
            metrics = mmengine.load(osp.join(output_dir, "metrics_summary.json"))
            metric_prefix = f"{result_name}_NuScenes_Tracking"
            for key in self.KEYS:
                detail[f"{metric_prefix}/{key}"] = metrics[key]
                for name in self.TRACKING_CLASSES:
                    detail[f"{metric_prefix}/{key}_{name}"] = metrics["label_metrics"][
                        key
                    ][name]
            # evaluate detection metrics
            det_metrics = super()._evaluate_single(result_path, classes, result_name)
            detail.update(det_metrics)
        except Exception as e:
            # log the error and continue
            print(f"Error occurs when evaluating {result_path}: {e}")

        return detail

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Use custom_collect_results rather than the regular collect_results

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f"{self.__class__.__name__} got empty `self.results`. Please "
                "ensure that the processed results are properly added into "
                "`self.results` in `process` method.",
                logger="current",
                level=logging.WARNING,
            )

        # use custom collect to handle multi-gpu evaluations, each gpu may have a
        # different number of frames
        results = custom_collect_results(
            self.results, 
            size, 
            self.collect_device, 
            tmpdir=self.collect_dir
        )

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {"/".join((self.prefix, k)): v for k, v in _metrics.items()}
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

        tracking (bool): if convert for tracking evaluation

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["bboxes_3d"]
    scores = detection["scores_3d"].numpy()
    if "track_scores" in detection.keys() and detection["track_scores"] is not None:
        scores = detection["track_scores"].numpy()
    labels = detection["labels_3d"].numpy()
    attrs = None
    if "attr_labels" in detection:
        attrs = detection["attr_labels"].numpy()

    if "forecasting" in detection.keys() and detection["forecasting"] is not None:
        forecasting = detection["forecasting"].numpy()
    else:
        forecasting = [None for _ in range(len(box3d))]

    if "track_ids" in detection.keys() and detection["track_ids"] is not None:
        track_ids = detection["track_ids"]
    else:
        track_ids = [None for _ in range(len(box3d))]

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    box_list = []

    assert isinstance(box3d, LiDARInstance3DBoxes)
    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesForecastingBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
            forecasting=forecasting[i],
            token=str(track_ids[i]),
        )
        box_list.append(box)
    return box_list, attrs


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, tracking=False):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info["lidar_points"]["lidar2ego"])
        box.rotate(pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        # filter out the classes not for tracking
        if (
            tracking
            and classes[box.label] not in NuScenesTrackingMetric.TRACKING_CLASSES
        ):
            continue
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info["ego2global"])
        box.rotate(pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list
