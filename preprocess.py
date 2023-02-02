import os
import torch
from torch.utils.data import DataLoader

from datasets.mini_imagenet import MiniImageNet
from datasets.tiered_imagenet import TieredImageNet
from datasets.cifarfs import CIFAR_FS
from datasets.fc100 import FC100
from datasets.omniglot import OmniglotDataset
from utils import seed_torch, set_gpu
from efnet import returnCusModel



def preprocess(args):

    if args.dataset == "mini":
        splits = ['train', 'val', 'test']
        model = returnCusModel(900).cuda()
        model.load_state_dict(torch.load('efficientnetb0-900-based-max.pth'))
        model.eval()
        with torch.no_grad():
            for split in splits:
                trainset = MiniImageNet(split=split, size=args.size)
                train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=0)

                datafile = {
                    'data': [],
                    'label': []
                }

                for i, batch in enumerate(train_loader, 1):
                    data, label = batch
                    f = model(data.cuda(), is_feat=True)
                    datafile['data'].append(f.squeeze(0).cpu())
                    datafile['label'].append(label.item())
                    print(i, "sample")
                torch.save(datafile, os.path.join(args.save_path, split+'_file.pt'))

    elif args.dataset == "tiered":
        splits = ['train', 'val', 'test']
        model = returnCusModel(392).cuda()
        model.load_state_dict(torch.load('efficientnetb0-392-based-max.pth'))
        model.eval()
        with torch.no_grad():
            for split in splits:
                data_set = TieredImageNet(split=split, size=args.size)
                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0)

                datafile = {
                    'data': [],
                    'label': []
                }

                for i, batch in enumerate(data_loader, 1):
                    data, label = batch
                    f = model(data.cuda(), is_feat=True)
                    datafile['data'].append(f.squeeze(0).cpu())
                    datafile['label'].append(label.item())
                    print(i, "sample")
                torch.save(datafile, os.path.join(args.save_path, split+'_file.pt'))
    
    elif args.dataset == "cifarfs":
        splits = ['train', 'val', 'test']
        model = returnCusModel().cuda()
        model.load_state_dict(torch.load('efficientnetb0-full-based-max.pth'))
        model.eval()
        with torch.no_grad():
            for split in splits:
                data_set = CIFAR_FS(split=split, size=args.size)
                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0)

                datafile = {
                    'data': [],
                    'label': []
                }

                for i, batch in enumerate(data_loader, 1):
                    data, label = batch
                    f = model(data.cuda(), is_feat=True)
                    datafile['data'].append(f.squeeze(0).cpu())
                    datafile['label'].append(label.item())
                    print(i, "sample")
                torch.save(datafile, os.path.join(args.save_path, split+'_file.pt'))
    
    elif args.dataset == "fc100":
        splits = ['train', 'val', 'test']
        model = returnCusModel().cuda()
        model.load_state_dict(torch.load('efficientnetb0-full-based-max.pth'))
        model.eval()
        with torch.no_grad():
            for split in splits:
                data_set = FC100(split=split, size=args.size)
                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0)

                datafile = {
                    'data': [],
                    'label': []
                }

                for i, batch in enumerate(data_loader, 1):
                    data, label = batch
                    f = model(data.cuda(), is_feat=True)
                    datafile['data'].append(f.squeeze(0).cpu())
                    datafile['label'].append(label.item())
                    print(i, "sample")
                torch.save(datafile, os.path.join(args.save_path, split+'_file.pt'))

    elif args.dataset == "omniglot":
        print("Omniglot...")
        model = returnCusModel().cuda()
        model.load_state_dict(torch.load('efficientnetb0-full-based-max.pth'))
        model.eval()
        with torch.no_grad():
            splits = ['trainval', 'test']
            for split in splits:
                datafile = {
                    'data': [],
                    'label': []
                }
                data_set = OmniglotDataset(mode=split)
                data_loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0)
                for i, data in enumerate(data_loader, 1):
                    x, y = data
                    x = torch.cat((x,x,x), 1)
                    x = model(x.cuda(), is_feat=True)
                    datafile['data'].append(x.cpu())
                    datafile['label'].append(y[0])
                    print(split, "sample", i, "with label", y[0].item())
                torch.save(datafile, os.path.join(args.save_path, split+"_file.pt"))
                print("Completed", split, "set...")
    else:
        print("Invalid dataset")

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--seed', type=int, default=2512)
    parser.add_argument('--save-path', default="./data/prefc100")
    parser.add_argument('--dataset', type=str, default='fc100', choices=['omniglot','mini','tiered','cifarfs','fc100'])
    parser.add_argument('--size', type=int, default=224)
    args = parser.parse_args()

    set_gpu(args.gpu)
    seed_torch(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    preprocess(args)
    print("Done...")