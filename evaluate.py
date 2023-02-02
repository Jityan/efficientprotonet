import argparse
import os.path as osp
import torch

from datasets.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from datasets.get_dataset import get_loader
from convnet import Convnet1d
from utils import set_gpu, Averager, count_acc, euclidean_metric, seed_torch, compute_confidence_interval

import datetime

def test(args):
    dataset = get_loader(args, 'test')
    sampler = CategoriesSampler(dataset.label,
                                args.test_batch, args.test_way, args.shot + args.test_query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=args.worker, pin_memory=True)

    model = Convnet1d().cuda()
    model.load_state_dict(torch.load(osp.join(args.save_path, 'max-acc.pth')))
    model.eval()

    ave_acc = Averager()
    acc_list = []

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.test_way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(args.shot, args.test_way, -1).mean(dim=0)
        p = x

        logits = euclidean_metric(model(data_query), p)

        label = torch.arange(args.test_way).repeat(args.test_query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        acc_list.append(acc*100)

        x = None; p = None; logits = None

    a, b = compute_confidence_interval(acc_list)
    return a, b

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/0')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2512)#2512
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini','tiered','cifarfs','fc100'])
    args = parser.parse_args()

    # fix seed
    seed_torch(args.seed)
    set_gpu(args.gpu)
    
    a, b = test(args)
    print("Final accuracy with 95% interval : {:.2f}±{:.2f}%".format(a, b))

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)
