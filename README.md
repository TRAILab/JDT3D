# [ECCV 2024] JDT3D: Addressing the Gaps in LiDAR-Based Tracking-by-Attention

[Brian Cheong](https://scholar.google.com/citations?user=q2x6PFwAAAAJ&hl=en&oi=sra), [Jiachen Zhou](https://scholar.google.ca/citations?user=QXfkGNwAAAAJ&hl=en), [Steven Waslander](https://www.trailab.utias.utoronto.ca/)

## Introduction

This is the official implementation of [JDT3D: Addressing the Gaps in LiDAR-Based Tracking-by-Attention](https://arxiv.org/abs/2407.04926).

<!-- If you find our code or paper useful, please cite by:
```tex

``` -->

### Prerequisites

* Docker
* NVIDIA GPU + CUDA CuDNN
* Download the [nuScenes](https://www.nuscenes.org/) dataset

### Pretrained models

| Model | Pretrained weights |
| --- | --- |
| BEVFusion | <https://github.com/TRAILab/JDT3D/releases/download/ckpt-upload/BEVFusion-L.pth> |
| JDT3D-f1 | <https://github.com/TRAILab/JDT3D/releases/download/ckpt-upload/jdt3d_f1.pth> |
| JDT3D-f3 | <https://github.com/TRAILab/JDT3D/releases/download/ckpt-upload/jdt3d_f3.pth> |

### Environment Setup

This code was tested using a docker environment to train and evaluate the models.

```bash
# Clone the repository
git clone https://github.com/TRAILab/JDT3D.git
cd JDT3D
# Build the docker image
make docker-build
# Run the docker container
make docker-dev
```

Modify the `Makefile` to set the correct paths to the nuScenes dataset (`DATA_ROOT_LOCAL`) and your directory for the job artifacts (`OUTPUT`).

### Preprocessing nuScenes

This may take a while to run, but the files generated only need to be generated once, and can be shared.

```bash
# run from inside the docker container
python tools/create_data.py nuscenes-tracking --root-path data/nuscenes --version v1.0-trainval --out-dir data/nuscenes --extra-tag nuscenes_track
```
We provide a zip of the preprocessed data files here: https://drive.google.com/file/d/1vVsD4Xg09lp2N67Q3pwieo-aVwQjEBdm/view?usp=sharing


### Training

To train JDT3D, pretrain the models using the f1 configuration and then train the models using the f3 configuration starting from the pretrained weights.

Update the paths of the pretrained weights in the configuration files using the `load_from` parameter.

Single GPU training, f1 training:

```bash
python tools/train projects/configs/tracking/jdt3d_f1.py --work-dir job_artifacts/jdt3d_f1
```

Single GPU training, f3 training:

```bash
python tools/train projects/configs/tracking/jdt3d_f3.py --work-dir job_artifacts/jdt3d_f3
```

Multi-GPU training, f1 training:

```bash
bash ./tools/dist_train.sh projects/configs/tracking/jdt3d_f1.py <num GPUs> --work-dir job_artifacts/jdt3d_f1
```

Multi-GPU training, f3 training:

```bash
bash ./tools/dist_train.sh projects/configs/tracking/jdt3d_f3.py <num GPUs> --work-dir job_artifacts/jdt3d_f3
```

Tensorboard training logs are saved to the `work-dirs` directory by default.

To speed up training, you can reduce the validation frequency by setting the `train_cfg.val_interval` parameter in the configuration file, found in Line 103 in `projects/configs/tracking/jdt3d_f1.py`. Because the f3 configuration inherits from the f1 configuration, the same parameter will change the `val_interval` for both configurations.

### Inference

Single GPU inference
  
```bash
python tools/test.py projects/configs/tracking/jdt3d_f3.py path/to/checkpoint --work-dir path/to/output
```

Multi-GPU inference

```bash
bash ./tools/dist_test.sh projects/configs/tracking/jdt3d_f3.py path/to/checkpoint <num GPUs> --work-dir path/to/output
```

## Acknowledgements

We thank the contributors to the following open-source projects. Our project would have been impossible without the work of these excellent researchers and engineers.

* 3D Detection. [MMDetection3d](https://github.com/open-mmlab/mmdetection3d), [DETR3D](https://github.com/WangYueFt/detr3d), [PETR](https://github.com/megvii-research/PETR).
* Multi-object tracking. [MOTR](https://github.com/megvii-research/MOTR), [MUTR3D](https://github.com/a1600012888/MUTR3D), [SimpleTrack](https://github.com/tusen-ai/SimpleTrack), [PF-Track](https://github.com/TRI-ML/PF-Track).
* End-to-end motion forecasting. [FutureDet](https://github.com/neeharperi/FutureDet).

In particular, we would like to thank the authors of [PF-Track](https://github.com/TRI-ML/PF-Track) for their detailed documentation and helpful structuring of their tracking-by-attention approach. We highly recommend any users of this repository to refer to their work and documentation for a more detailed understanding of how tracking-by-attention is implemented here.

## License

[![Creative Commons License](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](http://creativecommons.org/licenses/by-nc-sa/4.0/)
This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
