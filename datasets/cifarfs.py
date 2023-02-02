import os
import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data


class CIFAR_FS(Dataset):

    def __init__(self, dpath='c:\\Users\\User\\Desktop\\research_code\\data\\original\\cifar-fs', split='train', size=32):

        filepath = os.path.join(dpath, 'CIFAR_FS_' + split + ".pickle")
        datafile = load_data(filepath)
        
        data = datafile['data']
        label = datafile['labels']

        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        newlabel = []
        classlabel = 0
        for i in range(len(label)):
            if (i > 0) and (label[i] != label[i-1]):
                classlabel += 1
            newlabel.append(classlabel)

        self.data = data
        self.label = newlabel

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
                )
        ])
    
    def __len__(self):
        return len(self.data)
            
    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class CIFARFSFeatures(Dataset):

    def __init__(self, dpath="./data/precifarfs", split='train'):
        datafile = torch.load(os.path.join(dpath, split+"_file.pt"))

        self.data = datafile['data']
        self.label = datafile['label']
        self.n_classes = max(self.label) + 1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data1d = self.data[i].unsqueeze(0)
        return data1d, self.label[i]