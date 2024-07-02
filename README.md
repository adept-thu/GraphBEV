<div align="center">   

  # [ECCV2024] GraphBEV: Towards Robust BEV Feature Alignment for Multi-Modal 3D Object Detection 
</div>

<div align="center">
  <img src="fig/vis.gif" />
</div>

<div align="justify">  

This is the official repository of [**GraphBEV**](https://arxiv.org/abs/2403.11848). 

GraphBEV is designed to address the feature misalignment issue in previous BEV-based methods in real-world scenarios. In order to solve the problem of local misalignment, the **LocalAlign** module is introduced to obtain adjacent depth information through graphics, combined with explicit depth supervision from LiDAR to the camera. Then, the **GlobalAlign** module is proposed to encode the supervised depth and adjacent depth from LiDAR to the camera through dual depth encoding to generate a new reliable depth representation. 

Additionally, global misalignment issues are resolved by dynamically generating offsets. GraphBEV significantly outperforms BEVFusion on the nuScenes validation set, particularly in the presence of **noisy misalignment**.

:fire: Our work has been accepted by ECCV 2024!
</div>

------

<div align="justify">  

:fire: Our team focuses on Robustness of Autonomous driving, and we have summarized a  [**Survey**](https://arxiv.org/abs/2401.06542)  of Robustness.
Additionally, please pay attention to our another work on robustness, [**RoboFusion**](https://arxiv.org/abs/2401.03907), and its open-source [**repository**](https://github.com/adept-thu/RoboFusion).


:fire: Contributions:
* We propose a robust fusion framework, named GraphBEV, to address feature misalignment arising from projection errors between LiDAR and camera inputs.
* By deeply analyzing the fundamental causes of feature misalignment, we propose LocalAlign and GlobalAlign modules within our GraphBEV to address local misalignments from imprecise depth and global misalignments between LiDAR and camera BEV features.
* Extensive experiments validate the effectiveness of our GraphBEV, demonstrating competitive performance on nuScenes. Notably, GraphBEV maintains comparable performance across both clean settings and misaligned noisy conditions.

</div>

# Abstract

<div align="justify"> 

Integrating LiDAR and camera information into Bird's-Eye-View (BEV) representation has emerged as a crucial aspect of 3D object detection in autonomous driving. However, existing methods are susceptible to the inaccurate calibration relationship between LiDAR and the camera sensor. Such inaccuracies result in errors in depth estimation for the camera branch, ultimately causing misalignment between LiDAR and camera BEV features. In this work, we propose a robust fusion framework called GraphBEV. Addressing errors caused by inaccurate point cloud projection, we introduce a LocalAlign module that employs neighbor-aware depth features via Graph matching. Additionally, we propose a GlobalAlign module to rectify the misalignment between LiDAR and camera BEV features. Our GraphBEV framework achieves state-of-the-art performance, with an mAP of 70.1\%, surpassing BEVFusion by 1.6\% on the nuScnes validation set. Importantly, our GraphBEV outperforms BEVFusion by 8.3\% under conditions with misalignment noise.

</div>

# Method

<div align="center">
  <img src="fig/main.png" />
</div>

<div align="justify">

An overview of GraphBEV framework. The LiDAR branch almost follows the baselines (BEVfusion-MIT, TransFusion) to generate LiDAR BEV features. In the camera branch, first, we extract camera BEV features using proposed LocalAlign module that aim to addressing local misalignment due sensor calibration errors. Subsequently, we simulate the offset noisy of LiDAR and camera BEV features, followed by aligning global multi-modal features through learnable offsets. It is noteworthy that we only add offset noise to the GlobalAlign module during training to simulate global misalignment issues. Finally, we employ a dense detection head (TransFusion) to accomplish the 3D detection task.

</div>

# Model Zoo

* Results on nuScenes **val set**.

| Method | Modality | NDS⬆️ | mAP⬆️ | m BEV Map Seg.⬆️ | Config |
| :---: | :---: | :---: | :---: | :---: | :---: |
| BEVfusion-MIT | LC | 71.4 | 68.5 | 62.7 | [config](tools/cfgs/nuscenes_models/bevfusion_graph.yaml) |
| GraphBEV | LC | 72.9 | 70.1 | 63.3 | [config](tools/cfgs/nuscenes_models/bevfusion_graph_deformable.yaml) |

* Results on nuScenes **test set**.

| Method | Modality | NDS⬆️| mAP⬆️ |
| :---: | :---: | :---: | :---: |
| BEVfusion-MIT | LC | 72.9 | 70.2 |
| GraphBEV | LC | 73.6 | 71.7 |

* Results on nuScenes validation set under **noisy misalignment** setting.

| Method | Modality | NDS⬆️| mAP⬆️ | LT(ms)⬇️ |
| :---: | :---: | :---: | :---: | :---: |
| BEVfusion-MIT | LC | 65.7 | 60.8 | 132.9 |
| TransFusion | LC | 70.6 | 66.4 | 164.6 |
| GraphBEV | LC | 72.0 | 69.1 | 141.0 |

# Dataset Preparation

**NuScenes Dataset** : Please download the [official NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows:

```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

Install the `nuscenes-devkit` with version `1.0.5` by running the following command:

```bash
pip install nuscenes-devkit==1.0.5
```

Generate the data infos (for multi-modal setting) by running the following command (it may take several hours):

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam
```

# How to introduce misalignment noise into GraphBEV

GraphBEV is robust under various weather conditions. If you want to introduce misalignment noise into GraphBEV, please modify the following settings in [config](tools/cfgs/nuscenes_models/bevfusion_graph_deformable.yaml):
```
MODEL:
    ...
    VTRANSFORM:
        ...
        Noise: False
        K_graph: 25
...
```
Noisy Misalignment will be introduced when **Noise** is True, and **K_graph** represents the number of neighbor depths.

Whether Noisy Misalignment is introduced will be judged by the following [code](pcdet/models/view_transforms/depth_lss.py) :
```python
...
if not self.training:#test
    if self.noise:
        print("spatial_alignment_noise")
        lidar2image=self.spatial_alignment_noise(lidar2image,5)
        camera2lidar=self.spatial_alignment_noise(camera2lidar,5)
    else:
        print("clean")
...
```

Function **spatial_alignment_noise** is as following [code](pcdet/models/view_transforms/depth_lss.py) :
```python
def spatial_alignment_noise(self, ori_pose, severity):
    '''
    input: ori_pose 4*4
    output: noise_pose 4*4
    '''
    ct = [0.02, 0.04, 0.06, 0.08, 0.10][severity-1]*2
    cr = [0.002, 0.004, 0.006, 0.008, 0.10][severity-1]*2
    r_noise = torch.randn((3, 3), device=ori_pose.device)* cr
    t_noise = torch.randn((3), device=ori_pose.device) * ct
    ori_pose[..., :3, :3] += r_noise
    ori_pose[..., :3, 3]+= t_noise
    return ori_pose
```

# Requirements

All the codes are tested in the following environment:

* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.8+
* torch                     1.12.1+cu113
* torchaudio              0.12.1+cu113
* torchvision              0.13.1+cu113
* scipy                     1.10.1
* spconv-cu113              2.3.6

All codes are developed based on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) .

# Train and Inference

* Training is conducted on 8 NVIDIA GeForce RTX 3090 24G GPUs. 
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29535  train.py --launcher pytorch --batch_size 24  --extra_tag bevfusion_graph_deformable_result_scenes_K_graph8 --cfg_file cfgs/nuscenes_models/bevfusion_graph_deformable.yaml  --save_to_file 
```

* During inference, we remove Test Time Augmentation (TTA) data augmentation, and the batch size is set to 1 on an A100 GPU.
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29541 test.py --launcher pytorch --batch_size 1 --extra_tag bevfusion_graph_result_scenes_K_graph8 --cfg_file cfgs/nuscenes_models/bevfusion_graph.yaml --start_epoch 1 --eval_all --save_to_file --ckpt_dir ../output/nuscenes_models/bevfusion_graph/bevfusion_graph_result_scenes_K_graph8/ckpt
```

* All latency measurements are taken on the same workstation with an A100 GPU.

# Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{song2024graphbev,
  title={Graphbev: Towards robust bev feature alignment for multi-modal 3d object detection},
  author={Song, Ziying and Yang, Lei and Xu, Shaoqing and Liu, Lin and Xu, Dongyang and Jia, Caiyan and Jia, Feiyang and Wang, Li},
  journal={arXiv preprint arXiv:2403.11848},
  year={2024}
}

@article{song2024contrastalign,
  title={ContrastAlign: Toward Robust BEV Feature Alignment via Contrastive Learning for Multi-Modal 3D Object Detection},
  author={Song, Ziying and Jia, Feiyang and Pan, Hongyu and Luo, Yadan and Jia, Caiyan and Zhang, Guoxin and Liu, Lin and Ji, Yang and Yang, Lei and Wang, Li},
  journal={arXiv preprint arXiv:2405.16873},
  year={2024}
}

@article{song2023graphalign++,
  title={GraphAlign++: An accurate feature alignment by graph matching for multi-modal 3D object detection},
  author={Song, Ziying and Jia, Caiyan and Yang, Lei and Wei, Haiyue and Liu, Lin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}

@inproceedings{song2023graphalign,
  title={Graphalign: Enhancing accurate feature alignment by graph matching for multi-modal 3d object detection},
  author={Song, Ziying and Wei, Haiyue and Bai, Lin and Yang, Lei and Jia, Caiyan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3358--3369},
  year={2023}
}

@article{song2024robofusion,
  title={Robofusion: Towards robust multi-modal 3d obiect detection via sam},
  author={Song, Ziying and Zhang, Guoxing and Liu, Lin and Yang, Lei and Xu, Shaoqing and Jia, Caiyan and Jia, Feiyang and Wang, Li},
  journal={arXiv preprint arXiv:2401.03907},
  year={2024}
}



```


# Acknowledgement
Many thanks to these excellent open source projects:
- [BEVFusion-MIT](https://github.com/mit-han-lab/bevfusion) 
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- [TransFusion](https://github.com/XuyangBai/TransFusion/) 
