import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MiniImageNet(Dataset):

    def __init__(self, dpath="c:\\Users\\User\\Desktop\\research_code\\data\\original\\mini-imagenet\\", split='train', size=84):
        csv_path = osp.join(dpath, split + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(dpath, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class MiniImageNetFeatures(Dataset):

    def __init__(self, dpath="./data/premini", split='train'):
        datafile = torch.load(osp.join(dpath, split+"_file.pt"))

        self.data = datafile['data']
        self.label = datafile['label']
        self.n_classes = max(self.label) + 1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data1d = self.data[i].unsqueeze(0)
        return data1d, self.label[i]
