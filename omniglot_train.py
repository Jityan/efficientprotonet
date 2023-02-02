import argparse
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from datasets.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from datasets.omniglot import OmniglotFeatures
from convnet import Convnet1d1c
from utils import time_output, set_gpu, ensure_path, Averager, count_acc, euclidean_metric, seed_torch, compute_confidence_interval

import datetime
import time
import pytz

def main(args):
    ensure_path(args.save_path)

    trainset = OmniglotFeatures(split='trainval')
    train_sampler = CategoriesSampler(trainset.label, args.train_batch,
                                      args.train_way, args.shot + args.train_query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.worker, pin_memory=True)

    valset = OmniglotFeatures(split='test')
    val_sampler = CategoriesSampler(valset.label, args.val_batch,
                                    args.test_way, args.shot + args.train_query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=args.worker, pin_memory=True)

    model = Convnet1d1c().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
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

        tl, ta = train(args, model, train_loader, optimizer)

        lr_scheduler.step()

        vl, va = validate(args, model, val_loader)

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')
            best_epoch = epoch

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        time2 = time.time()

        if args.detail:
            print('Epoch {}/{}, train loss={:.4f} - acc={:.4f} - val loss={:.4f} - acc={:.4f} - max acc={:.4f} [{} total {}]'.format(
                epoch, args.max_epoch, tl, ta, vl, va, trlog['max_acc'],
                datetime.datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime("%H:%M"),
                time_output(time2-time1)))

        if epoch == args.max_epoch:
            print("Best Epoch is {} with acc={:.4f}...".format(best_epoch, trlog['max_acc']))
            print("---------------------------------------------------")

    return trlog['train_acc'], trlog['val_acc'], best_epoch

def train(args, model, train_loader, optimizer):
    model.train()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = args.shot * args.train_way
        data_shot, data_query = data[:p], data[p:] # datashot (30, 3, 84, 84)
            
        proto = model(data_shot) # (30, 1600)
        proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

        label = torch.arange(args.train_way).repeat(args.train_query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        proto = None; logits = None; loss = None
    return tl.item(), ta.item()

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

def test(args):
    dataset = OmniglotFeatures(split='test')
    sampler = CategoriesSampler(dataset.label,
                                args.test_batch, args.test_way, args.shot + args.test_query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=args.worker, pin_memory=True)

    model = Convnet1d1c().cuda()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--shot', type=int, default=1) # shot
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--train-query', type=int, default=15)
    parser.add_argument('--test-query', type=int, default=15)
    parser.add_argument('--save-path', default='./save/omniglot-t')
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--train-batch', type=int, default=100)
    parser.add_argument('--val-batch', type=int, default=400)
    parser.add_argument('--test-batch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--detail', default=False, action='store_true')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    start_time = datetime.datetime.now()
    # fix seed
    seed_torch(args.seed)
    set_gpu(args.gpu)
    #
    if args.mode == 0:
        train_acc, val_acc, _ = main(args)
        # print graph for accuracy
        plt.figure(figsize=(10,5))
        plt.title("Training Accuracy")
        plt.plot(train_acc, label="Training")
        plt.plot(val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    else:
        a, b = test(args)
        print("Average accuracy with 95% interval : {:.2f}±{:.2f}".format(a, b))

    end_time = datetime.datetime.now()
    print("Total executed time :", end_time - start_time)

