from datasets.mini_imagenet import MiniImageNet, MiniImageNetFeatures
from datasets.tiered_imagenet import TieredImageNet, TieredImageNetFeatures
from datasets.fc100 import FC100, FC100Features
from datasets.cifarfs import CIFAR_FS, CIFARFSFeatures

def get_loader(args, split, features=True):
    names = ['mini', 'tiered', 'cifarfs', 'fc100', 'cifarfsaug']
    splits = ['train', 'val', 'test']

    if args.dataset not in names or split not in splits:
        print("Wrong name or split")
        exit()
    
    if args.dataset == "mini" and features:
        dataset = MiniImageNetFeatures(split=split)
    elif args.dataset == "tiered" and features:
        dataset = TieredImageNetFeatures(split=split)
    elif args.dataset == "cifarfs" and features:
        dataset = CIFARFSFeatures(split=split)
    elif args.dataset == "fc100" and features:
        dataset = FC100Features(split=split)
    elif args.dataset == "mini" and not features:
        dataset = MiniImageNet(split=split)
    elif args.dataset == "tiered" and not features:
        dataset = TieredImageNet(split=split)
    elif args.dataset == "cifarfs" and not features:
        dataset = CIFAR_FS(split=split)
    elif args.dataset == "fc100" and not features:
        dataset = FC100(split=split)
    return dataset