# AD4NS Dataset

#### Introduction

This is a new dataset for Neural Symbolic, based on one famous dataset for Autonomous Driving, called [NuScenes](https://www.nuscenes.org/nuscenes) 

This dataset describes a simplified task in Autonomous Driving, which takes a camera-captured image as input, going through a 3D objection detection module as the **neural part**, following a path planning module as the **symbolic part**, and finally gives a optimized path to follow as output.

We believe this is a non-trivial example for Neural Symbolic research, and may help to handle the complicated Neural Symbolic hybrid software in the future.

#### Usage

* git clone this repository

* The *'data'* directory consists of three sub-folders, including:
  * *'algorithms'*: It contains the codes for path planning, e.g. 'astar.py' defines a class of A* algorithm
  * *'models'*: It should contain the models for 3D objection detection, e.g. *'FCOS3D.pth'* saves the networks and parameters for [FCOS3D](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcos3d) model (click [this](https://box.nju.edu.cn/f/52ac9b2576ae4594b536/?dl=1) to download the models folder and put it in the root *'data/'*)
  * *'datasets'*: It should contains the datasets for this detection&planning task, e.g. *'nuscenes-mini'* saves the 105 images and their corresponding input/output text files(click [this](https://box.nju.edu.cn/f/619ba363a46740d69888/?dl=1) to download the datasets folder and put it in the root *'data/'*)
* The *'src'* directory contains a *'utils.py'*, which defines two classes (AD4NSDataset and AD4NSDataLoader) to load and use data, including the datasets, models, and algorithms
  * You can know more about how to use them in *'tutorial.ipynb'*

