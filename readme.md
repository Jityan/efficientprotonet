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
    - `datasets/omniglot.py` (only change the path in **OmniglotFeatures** class for extracted features)

2. [Optional] Download the original datasets and put them into corresponding folders:<br/>
    - ***mini*ImageNet**: download from https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE and put in `data/miniImageNet/images` folder.

    - ***tiered*ImageNet**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/tieredImageNet` folder.

    - **CIFARFS**: download from [MetaOptNet](https://github.com/kjunelee/MetaOptNet) and put in `data/cifarfs` folder.

    - **FC100**: download from [MTL](https://github.com/yaoyao-liu/meta-transfer-learning), extract them into train, val, and test folders and put in `data/fc100` folder.

    - **Omniglot**: the download for Omniglot is automatic when the script executed.

## Extract Features
1. The extracted features can be downloaded from [here](https://drive.google.com/drive/folders/1xsFs5n_K12l-b2S0rDiprOYPRTrVK90d?usp=share_link) or

2. [Optional] The pretrained weights for EfficientNet-B0 can be downloaded from here: [*mini*ImageNet](https://drive.google.com/file/d/1oKQVT8uRVb0L0mgwUHj89GyVzAHm7OFs/view?usp=share_link), [*tiered*ImageNet](https://drive.google.com/file/d/1OSI6zNsa82d5NzA8asJAZFJSEcpn8txt/view?usp=sharing), and [other datasets](https://drive.google.com/file/d/1G0Q-3gTlvDEBULdbrOoasgLmGFTqxSfz/view?usp=share_link). Run the below command extract the feature vectors, NAME decide the dataset and PATH define the location to store the extracted features. <br/>
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
To train on *N*-way 5-shot Omniglot:<br/>
```
python omniglot_train.py --shot 5 --mode 0
python omniglot_train_distill.py --shot 5 --mode 0
```
To evaluate on *N*-way 5-shot Omniglot:<br/>
```
python omniglot_train_distill.py --shot 5 --mode 1
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
  author = {Jit Yan Lim and Kian Ming Lim and Shih Yin Ooi and Chin Poo Lee}
}
```

## Contacts
For any questions, please contact: <br/>

Jit Yan Lim (jityan95@gmail.com) <br/>
Kian Ming Lim (Kian-Ming.Lim@nottingham.edu.cn)

## Acknowlegements
This repo is based on **[Prototypical Networks](https://github.com/yinboc/prototypical-network-pytorch)**, **[EfficientNet](https://github.com/narumiruna/efficientnet-pytorch)**, **[MetaOptNet](https://github.com/kjunelee/MetaOptNet)**, and **[MTL](https://github.com/yaoyao-liu/meta-transfer-learning)**.
