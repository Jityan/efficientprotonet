import os.path as osp
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class FC100(Dataset):

    def __init__(self, dpath="c:\\Users\\User\\Desktop\\research_code\\data\\original\\FC100\\", split='train', size=32):
        # Set the path according to train, val and test
        if split=='train':
            THE_PATH = osp.join(dpath, 'train')
            label_list = os.listdir(THE_PATH)
        elif split=='test':
            THE_PATH = osp.join(dpath, 'test')
            label_list = os.listdir(THE_PATH)
        elif split=='val':
            THE_PATH = osp.join(dpath, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Wrong setname.') 

        # Generate empty list for data and label           
        data = []
        label = []

        # Get folders' name
        folders = [osp.join(THE_PATH, the_label) for the_label in label_list if os.path.isdir(osp.join(THE_PATH, the_label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        # Set data, label and class number to be accessable from outside
        self.data = data
        self.label = label
        self.num_class = len(set(label))

        #print("Label :", len(self.label))
        #print("Class :", self.num_class)

        # Transformation
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
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class FC100Features(Dataset):

    def __init__(self, dpath="./data/prefc100", split='train'):
        datafile = torch.load(osp.join(dpath, split+"_file.pt"))

        self.data = datafile['data']
        self.label = datafile['label']
        self.n_classes = max(self.label) + 1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data1d = self.data[i].unsqueeze(0)
        return data1d, self.label[i]

