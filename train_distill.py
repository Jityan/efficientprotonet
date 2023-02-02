import argparse
import os.path as osp
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.get_dataset import get_loader
from datasets.samplers import CategoriesSampler
from convnet import Convnet1d, DistillKL
from utils import time_output, set_gpu, ensure_path, Averager, count_acc, euclidean_metric, seed_torch, compute_confidence_interval

import copy
import datetime
import time
import pytz

def save_model(name, model):
    torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))

def train(args, train_loader, model, optimizer, teacher, kd_loss):
    model.train()
    teacher.eval()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:]
        label = torch.arange(args.train_way).repeat(args.train_query)
        label = label.type(torch.cuda.LongTensor)

        with torch.no_grad():
            tproto = teacher(data_shot)
            tproto = tproto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            tlogits = euclidean_metric(teacher(data_query), tproto)

        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        # knowledge distill
        kdloss = kd_loss(logits, tlogits)
        total_loss = loss + args.alpha * kdloss

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
            
        proto = None; logits = None; loss = None
        tproto = None; tlogits = None; ft = None; fs = None

    return tl.item(), ta.item()

def main(args):
    ensure_path(args.save_path)

    trainset = get_loader(args, 'train')
    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                      args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.worker, pin_memory=True)

    valset = get_loader(args, 'val')
    val_sampler = CategoriesSampler(valset.label, args.val_batch,
                                    args.test_way, args.shot + args.train_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)

    teacher = Convnet1d().cuda()
    teacher.load_state_dict(torch.load(osp.join(args.pretrain_path, 'max-acc.pth')))
    model = copy.deepcopy(teacher)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion_kd = DistillKL(args.temperature).cuda()

    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0
    best_epoch = 0

    for epoch in range(1, args.max_epoch + 1):
        time1 = time.time()

        tl, ta = train(args, train_loader, model, optimizer, teacher, criterion_kd)

        lr_scheduler.step()

        vl, va = validate(args, model, val_loader)

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc', model)
            best_epoch = epoch

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last', model)
        time2 = time.time()
        if args.detail:
            print('Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f} - max acc={:.4f} [{} total {}]'.format(
                epoch, args.max_epoch, tl, ta, vl, va, trlog['max_acc'],
                datetime.datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime("%H:%M"),
                time_output(time2-time1)))

    if args.detail:
        print("---------------------------------------------------")

    return trlog['train_acc'], trlog['val_acc'], best_epoch

def validate(args, model, val_loader):
    model.eval()

    vl = Averager()
    va = Averager()

    for i, batch in enumerate(val_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

        label = torch.arange(args.test_way).repeat(args.train_query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        vl.add(loss.item())
        va.add(acc)
            
        proto = None; logits = None; loss = None

    vl = vl.item()
    va = va.item()
    return vl, va

def test(args, role):
    seed_torch(args.seed)
    dataset = get_loader(args, 'test')
    sampler = CategoriesSampler(dataset.label,
                                args.test_batch, args.test_way, args.shot + args.test_query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=args.worker, pin_memory=True)

    model = Convnet1d().cuda()
    if role == 's':
        model.load_state_dict(torch.load(osp.join(args.save_path, 'max-acc.pth')))
    else:
        model.load_state_dict(torch.load(osp.join(args.pretrain_path, 'max-acc.pth')))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1) # shot
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30) # train way
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/g1')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--val-batch', type=int, default=400)
    parser.add_argument('--test-batch', type=int, default=2000)
    parser.add_argument('--test-query', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wd', type=float, default=0.1)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2512)
    parser.add_argument('--pretrain-path', default='./save/g0')
    parser.add_argument('--temperature', type=int, default=4)
    parser.add_argument('--loss-gamma', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini','tiered','cifarfs','fc100'])
    parser.add_argument('--detail', default=True, action='store_true')
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    best_epoch = 0

    # fix seed
    seed_torch(args.seed)
    set_gpu(args.gpu)

    train_acc, val_acc, best_epoch = main(args)

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)

    # print graph for accuracy
    plt.figure(figsize=(10,5))
    plt.title("Training Accuracy")
    plt.plot(train_acc, label="Training")
    plt.plot(val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

