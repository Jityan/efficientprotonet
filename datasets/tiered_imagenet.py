import os
import pickle
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TieredImageNet(Dataset):

    def __init__(self, dpath='c:\\Users\\User\\Desktop\\research_code\\data\\original\\tiered-imagenet-kwon\\', split='train', size=84):
        split_tag = split
        data = np.load(os.path.join(
                dpath, '{}_images.npz'.format(split_tag)),
                allow_pickle=True)['images']
        data = data[:, :, :, ::-1]

        with open(os.path.join(
                dpath, '{}_labels.pkl'.format(split_tag)), 'rb') as f:
            label = pickle.load(f)['labels']

        data = [Image.fromarray(x) for x in data]

        min_label = min(label)
        label = [x - min_label for x in label]

        self.data = data
        self.label = label
        self.n_classes = max(self.label) + 1

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]


class TieredImageNetFeatures(Dataset):

    def __init__(self, dpath="./data/pretiered", split='train'):
        datafile = torch.load(os.path.join(dpath, split+"_file.pt"))

        self.data = datafile['data']
        self.label = datafile['label']
        self.n_classes = max(self.label) + 1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data1d = self.data[i].unsqueeze(0)
        return data1d, self.label[i]
