
# Cylindrical 3D Transformer Network for 3D LiDAR Segmentation

## Installation

### pull repo
```
git clone https://github.com/nerovalerius/AttentiveCylinder3D.git
```

### install anaconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
sh ./Miniconda3-py39_4.12.0-Linux-x86_64.sh
```

### activate conda
Either restart terminal or do:
```source ~/.bashrc```

### create conda environment
```
conda create -n attentive_cylinder_3d
conda activate attentive_cylinder_3d
```

### update conda
```conda update -n base -c defaults conda```


### install python and packages
conda install python=3.9.2 numpy tqdm pyyaml numba strictyaml -c conda-forge


### install cuda (warning: not neccessary if WSL 2.0 is used)
```
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
```

### install torch and cudatoolkit 11.6
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

### install spconv for cuda 11.6
Version "cu116" was not available at the time, however cu114 also works.
```
pip install spconv-cu114
```

### install torch-spares and scatter for cuda 11.6
```
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html
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