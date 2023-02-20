#!/bin/bash
sudo docker run --rm -it --gpus '"device=0"' --cpuset-cpus="0-15" -m 48g --shm-size 48g -v /home/mta/mdilab/lidar_semantic_segmentation/datasets/:/workspace/datasets_mta -v /home/asams/projects/masterthesis/:/workspace/ -p 12211:8888 --name attentivecylinder3d container.salzburgresearch.at/mobility/projects/mdi-lab/attentivecylinder3d
