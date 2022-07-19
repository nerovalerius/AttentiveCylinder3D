
# Cylindrical 3D Transformer Network for 3D LiDAR Segmentation

## Installation

### Requirements
### pull repo
```
git clone https://github.com/xinge008/Cylinder3D.git
```

### create environment
```
conda create -n attentive_cylinder_3d
conda activate attentive_cylinder_3d
pip install tqdm pyyaml numba spconv
```

### install cuda
```
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run --silent --driver
sudo sh cuda_11.7.0_515.43.04_linux.run
```

### install torch and cudatoolkit
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### install correct spconv
```
pip install spconv-cu113 -i https://pypi.python.org/simple
```

### install correct torch-spares and scatter
```
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0%2Bcu113.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0%2Bcu113.html
```

### install nuscenes devkit
- [nuScenes-devkit](https://github.com/nutonomy/nuscenes-devkit) (optional for nuScenes)


## Data Preparation

### SemanticKITTI
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
    ├──sequences
        ├── 00/           
        │   ├── velodyne/	
        |   |	├── 000000.bin
        |   |	├── 000001.bin
        |   |	└── ...
        │   └── labels/ 
        |       ├── 000000.label
        |       ├── 000001.label
        |       └── ...
        ├── 08/ # for validation
        ├── 11/ # 11-21 for testing
        └── 21/
	    └── ...
```

### nuScenes
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
		├──v1.0-trainval
		├──v1.0-test
		├──samples
		├──sweeps
		├──maps

```

## Usage


# train network
```
sh train.sh
```


## Training

### Training semanticKITTI
1. modify ```config/semantickitti.yaml``` with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running ```sh train.sh```

### Training nuScenes
Please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)

### Pretrained Models for the original Cylinder3D
-- SemanticKITTI [LINK1](https://drive.google.com/file/d/1q4u3LlQXz89LqYW3orXL5oTs_4R2eS8P/view?usp=sharing) or [LINK2](https://pan.baidu.com/s/1c0oIL2QTTcjCo9ZEtvOIvA) (access code: xqmi)

-- For nuScenes dataset, please refer to [NUSCENES-GUIDE](./NUSCENES-GUIDE.md)

## Semantic segmentation demo for a single sequence of lidar scans
Set the correct model folders for save and load inside ```config/semantickitti.yaml```.

```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER
```
If you want to validate with your own datasets, you need to provide labels.
--demo-label-folder is optional
```
python demo_folder.py --demo-folder YOUR_FOLDER --save-folder YOUR_SAVE_FOLDER --demo-label-folder YOUR_LABEL_FOLDER
```
# example usage

```
python demo_folder.py --demo-folder ../dataset/sequences/00/velodyne/ --demo-label-folder ../dataset/sequences/00/labels/ --save-folder save_folder/ 
```

## Rights
This network builds upon [Cylinder3D](https://github.com/xinge008/Cylinder3D) from Zhu et.al.

If you find our their useful in your research, please consider citing their [paper](https://arxiv.org/pdf/2011.10033):
```
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}
```