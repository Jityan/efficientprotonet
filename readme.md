# Efficient-PrototypicalNet with self knowledge distillation for few-shot learning

This repository contains the **pytorch** code for the paper: "[Efficient-PrototypicalNet with self knowledge distillation for few-shot learning](https://doi.org/10.1016/j.neucom.2021.06.090)" Jit Yan Lim, Kian Ming Lim, Shih Yin Ooi, Chin Poo Lee

## Environment
The code is tested on Windows 10 with Anaconda3 and following packages:
- python 3.7.4
- pytorch 1.3.1

## Preparation
1. Change the path value in the following files to yours:
    - `datasets/mini_imagenet.py`
    - `datasets/tiered_imagenet.py`
    - `datasets/cifarfs.py`
    - `datasets/fc100.py`

2. [Optional] Download the original datasets and put them into corresponding folders:<br/>
    - ***mini*ImageNet**: download from https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE and put in `data/miniImageNet/images` folder.

    - ***tiered*ImageNet**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tieredImageNet` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tieredImageNet` and `data/cifarfs` folder.

    - **FC100**: download from [MTL](https://github.com/yaoyao-liu/meta-transfer-learning), extract them into train, val, and test folders and put in `data/fc100` folder.

## Extract Features
1. The extracted features can be downloaded from here : (TBA) or

2. [Optional] The pretrained weights for EfficientNet-B0 can be downloaded from here: (TBA) and run the below command extract the feature vectors, NAME decide the dataset and PATH define the location to store the extracted features. <br/>
    ```
    python preprocess.py --dataset NAME --save-path PATH
    ```

## Experiments
To train on 30-way 1-shot miniImageNet:<br/>
```
python train.py --dataset mini --train-way 30 --shot 1 --save-path ./save/mini_30w1s
```
To train the 30-way 1-shot miniImageNet with SSKD:<br/>
```
python train_distill.py --dataset mini --train-way 30 --shot 1 --pretrain-path ./save/mini_30w1s --save-path ./save/mini_30w1s_distill
```
To evaluate on 5-way 1-shot miniImageNet:<br/>
```
python evaluate.py --dataset mini --test-way 5 --shot 1 --save-path ./save/mini_30w1s_distill
```

## Citation
If you find this repo useful for your research, please consider citing the paper:
```
@article{LIM2021327,
  title = {Efficient-PrototypicalNet with self knowledge distillation for few-shot learning},
  journal = {Neurocomputing},
  volume = {459},
  pages = {327-337},
  year = {2021},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2021.06.090},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231221010262},
  author = {Jit Yan Lim and Kian Ming Lim and Shih Yin Ooi and Chin Poo Lee},
  keywords = {Few-shot learning, Meta learning, EfficientNet, Prototypical network, Transfer learning, Knowledge distillation},
  abstract = {The focus of recent few-shot learning research has been on the development of learning methods that can quickly adapt to unseen tasks with small amounts of data and low computational cost. In order to achieve higher performance in few-shot learning tasks, the generalizability of the method is essential to enable it generalize well from seen tasks to unseen tasks with limited number of samples. In this work, we investigate a new metric-based few-shot learning framework which transfers the knowledge from another effective classification model to produce well generalized embedding and improve the effectiveness in handling unseen tasks. The idea of our proposed Efficient-PrototypicalNet involves transfer learning, knowledge distillation, and few-shot learning. We employed a pre-trained model as a feature extractor to obtain useful features from tasks and decrease the task complexity. These features reduce the training difficulty in few-shot learning and increase the performance. Besides that, we further apply knowledge distillation to our framework and achieve extra performance improvement. The proposed Efficient-PrototypicalNet was evaluated on five benchmark datasets, i.e., Omniglot, miniImageNet, tieredImageNet, CIFAR-FS, and FC100. The proposed Efficient-PrototypicalNet achieved the state-of-the-art performance on most datasets in the 5-way K-shot image classification task, especially on the miniImageNet dataset.}
}
```

## Contacts
For any questions, please contact: <br/>

Jit Yan Lim (1141124378@student.mmu.edu.my) <br/>
Kian Ming Lim (kmlim@mmu.edu.my)

## Acknowlegements
This repo is based on **[Prototypical Networks](https://github.com/yinboc/prototypical-network-pytorch)**, **[EfficientNet](https://github.com/narumiruna/efficientnet-pytorch)**, **[MetaOptNet](https://github.com/kjunelee/MetaOptNet)**, and **[MTL](https://github.com/yaoyao-liu/meta-transfer-learning)**.