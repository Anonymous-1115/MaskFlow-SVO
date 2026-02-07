# Towards robust visual odometry in dynamic environments: A hybrid approach with confidence-guided masking

## Note
This work is currently under peer review. To comply with the double-blind review policy and ensure the reproducibility of our key results, only a subset of the code is publicly available at this stage. The full source code will be completely open-sourced under the MIT License once the paper is accepted for publication.
## Abstract
Dynamic visual odometry is critical for safe autonomous driving and robot navigation. Although traditional geometry-based dynamic visual odometry methods can achieve high accuracy, they often lack robustness in low-texture dynamic environments. 
In contrast, learning-based methods exhibit better robustness under such conditions but still fall short in accuracy.
To bridge this gap, we propose a novel dynamic stereo visual odometry framework that integrates end-to-end learning with multi-frame geometric optimization. Specifically, 
we design a lightweight Transformer-based network that leverages optical flow and disparity to predict metrically scaled ego-motion. A confidence-guided dynamic mask, derived from both network predictions and semantic priors, enables the network to focus on reliable static regions and suppress the influence of moving objects. Pose estimation is then performed based on a masked attention mechanism.
To further enhance accuracy, we introduce a sliding window optimization that refines the initial coarse pose. By leveraging the dynamic masks, we establish reliable matches and optimize the poses using  a combination of reprojection factor, an odometry factor, and a scale factor. 
Extensive experiments on diverse public datasets and a self-collected dataset show that our method achieves state-of-the-art accuracy and robustness in dynamic environments.

![Overview](./assets/overview.png)

## Installation
```
git clone https://github.com/Anonymous-1115/MaskFlow-SVO.git
cd ./MaskFlow-SVO
conda create -n maskflow python=3.10
conda activate maskflow
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
```

### pytorch3d
Move to https://anaconda.org/pytorch3d/pytorch3d/files, download the right one, and run 
```
conda install pytorch3d-0.7.8-py310_cu121_pyt241.tar.bz2
```
(It seems acceptable with minor version mismatches.)

### g2o-python
```
cd ./thirdparty/g2o-python
python -m pip install -v . -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
cd ../../
```

## KITTI-Tracking
### Dataset
1. Download left/right image, calibration files and GPS/IMU data from https://www.cvlibs.net/datasets/kitti/eval_tracking.php
2. Generate Ground-truth poses from GPS/IMU data:
```
python -m datasets.KITTI-dev-tools.oxts_to_pose --basedir <YOUR PLACE>/data_tracking_oxts --output <YOUR PLACE>/data_tracking_poses  
```
3. Place them in the same folder, this root folder can be set at [configs/kitti_tracking.yaml](./configs/kitti_tracking.yaml), the structure is like:
```
.
├── data_tracking_calib
├── data_tracking_image_2
├── data_tracking_image_3
├── data_tracking_oxts
└── data_tracking_poses
```
4. Generate cache (flow, disparity, prior mask), take sequence 02 for example: 
```
python -m datasets.generate_cache.cache_kitti_tracking --seq 02  
```

### Evaluation
Run MaskFlow-SVO. You can also manually specify the running sequence in [configs/kitti_tracking.yaml](./configs/kitti_tracking.yaml).
```
python -m main --dataset kitti_tracking --seq 02
```
Only run MaskFlow-SVO-Net to trade-off speed and accuracy.
```
python -m main --dataset kitti_tracking --seq 02 --no_ba
```

## KITTI-Odometry
### Dataset
1. Download color image, calibration files and ground-truth pose from https://www.cvlibs.net/datasets/kitti/eval_odometry.php
2. Place them in the same folder, this root folder can be set at [configs/kitti.yaml](./configs/kitti.yaml), the structure is like:
```
.
├── data_odometry_calib
├── data_odometry_color
└── data_odometry_poses
```
3. Generate cache (flow, disparity, prior mask), take sequence 07 for example: 
```
python -m datasets.generate_cache.cache_kitti --seq 07  
```

### Evaluation
Run MaskFlow-SVO. You can also manually specify the running sequence in [configs/kitti.yaml](./configs/kitti.yaml).
```
python -m main --dataset kitti --seq 07
```
Only run MaskFlow-SVO-Net to trade-off speed and accuracy.
```
python -m main --dataset kitti --seq 07 --no_ba
```

## License
This software is MIT licensed.

Copyright (c) [2026] Anonymous Authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.