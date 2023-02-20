
# Attentive Cylindrical 3D Transformer Network for 3D LiDAR Segmentation
This network builds upon [Cylinder3D](https://github.com/xinge008/Cylinder3D) from Zhu et.al.

## I - Installation and usage w/o docker

### pull repo
```
git clone https://github.com/nerovalerius/AttentiveCylinder3D.git
```

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

## II - Installation docker

### build docker
I strongly recommend that you use docker. This docker mounts a workspace where this git repo should be cloned in.
In comparison to the package versions described in this readme, the docker uses some newer versions.
Adapt your workspace inside ```build_docker.sh ``` and then run ```sh build_docker.sh ```.

### run docker
Simply run ```./start_docker.sh``` to get the docker up and running. Afterwards, inside the docker, install the Minkowski Engine with ```./install_minkowsk.sh```.
Then you could either start jupyter notebooks with ```./start_jupyter.sh```, or convert the notebook to a python file for training without jupyter:
````jupyter nbconvert train_attentivecylinder3d.ipynb --to python```. 

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

# Usage

## (Optional) - start docker interactively first
Use ```sh run_docker``` to start an interactive docker container.

## (Optional) - start jupyter notebook
There is also a script to start a jupyter lab instance on port 12212. Just run ```sh start_jupyter.sh``` inside your -it docker workspace/attentivecylinder3d/ and open your (remote browser). 
The main file to work with is ```train_cylinder_asym_jupyter.ipynb```.

# Train network (either inside the interactive docker or without docker)
```python train_attentivecylinder3d.py```

## Configuration for different datasets

### Training semanticKITTI
1. modify ```config/semantickitti.yaml``` with your custom settings. We provide a sample yaml for SemanticKITTI
2. train the network by running ```python train_attentivecylinder3d.py```

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
# Inference - example usages

```
python demo_folder.py --demo-folder ../dataset/sequences/00/velodyne/ --demo-label-folder ../dataset/sequences/00/labels/ --save-folder save_folder/ 
python demo_folder.py --demo-folder /home/nero/master/dataset/sequences/00/velodyne/ --save-folder save_folder/
python demo_folder.py --demo-folder /home/nero/semanticKITTI/dataset/sequences/00/velodyne/ --save-folder save_folder/ --demo-label-folder home/nero/semanticKITTI/dataset/sequences/00/labels/
```

# Get statistics out of the dataset
```git clone https://github.com/PRBonn/semantic-kitti-api.git```

Run ./content.py --directory dataset/ to achieve statistics about the labels inside the dataset.
Adapt the semantic-kitti-api/config/semantic-kitti.yaml file beforehand or use the sbld.yaml file inside this repo under: attentivecylinder3d/config/label_mapping/sbld.yaml. The train/test split folders must match the folder structure of train/test if you have one.


## Rights
This network mainly builds upon [Cylinder3D](https://github.com/xinge008/Cylinder3D) from Zhu et.al.

If you find our their useful in your research, please consider citing their [paper](https://arxiv.org/pdf/2011.10033):
```
@article{zhu2020cylindrical,
  title={Cylindrical and Asymmetrical 3D Convolution Networks for LiDAR Segmentation},
  author={Zhu, Xinge and Zhou, Hui and Wang, Tai and Hong, Fangzhou and Ma, Yuexin and Li, Wei and Li, Hongsheng and Lin, Dahua},
  journal={arXiv preprint arXiv:2011.10033},
  year={2020}
}
```

Furthermore, the transformer blocks from [CodedVTR](https://github.com/A-suozhang/CodedVTR) are used in this work, which is based on [SpatioTemporalSegmentation-ScanNet](https://github.com/chrischoy/SpatioTemporalSegmentation-ScanNet).

@inproceedings{zhao2022codedvtr,
  title={CodedVTR: Codebook-based Sparse Voxel Transformer with Geometric Guidance},
  author={Zhao, Tianchen and Zhang, Niansong and Ning, Xuefei and Wang, He and Yi, Li and Wang, Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1435--1444},
  year={2022}
}


