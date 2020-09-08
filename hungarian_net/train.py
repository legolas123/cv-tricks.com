import argparse
import os
import shutil
import time
from models import GraphModel, edge_loss
from dataset import HungarianDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import logging
from torch_geometric.data import DataLoader

"""
Running Commands
CUDA_VISIBLE_DEVICES=1 python3 train.py
"""
"""
Installations:
    pip3 install torch_geometric
    pip install lapsolver
    export CUDA=cu92
    pip3 install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.5.0.html
"""


lr = 1e-1
weight_decay = 1e-4
batch_size = 64
print_freq = 20
num_train_samples = 10000
num_test_samples = 1024
num_epochs = 50

def main():

    model = GraphModel()

    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    dataset = HungarianDataset(num_train_samples, mode="train")
    train_loader = DataLoader(
                    dataset, batch_size=batch_size,
                    num_workers=4, pin_memory=True, drop_last=True)

    test_dataset = HungarianDataset(num_test_samples, mode="test")
    test_loader = DataLoader(
                    test_dataset, batch_size=batch_size,
                    num_workers=4, pin_memory=True, drop_last=True)

    for epoch in range(0,num_epochs):
        adjust_learning_rate(optimizer, epoch)
        validate(test_loader, model, epoch)
        # train for one epoch
        train(train_loader, model, optimizer, epoch)
        
def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    for i, graph in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph.to(torch.device("cuda"))
        preds = model(graph)
        loss = edge_loss(preds, graph.labels)
        # if epoch > 10:
        #     import pdb; pdb.set_trace()
        #     print(torch.sigmoid(preds[-1])[graph.labels>0.5])
        #     print(torch.sigmoid(preds[-1])[graph.labels<0.5][:100])
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        if i % print_freq == 0:
            
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))

def validate(test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top_pos = AverageMeter()
    top_neg = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()
    for i, graph in enumerate(test_loader):
        data_time.update(time.time() - end)
        graph.to(torch.device("cuda"))
        preds = model(graph)
        loss = edge_loss(preds, graph.labels)
        losses.update(loss.item(), batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        acc_pos, num_pos, acc_neg, num_neg = accuracy(preds[-1], graph.labels)
        top_pos.update(acc_pos, num_pos)
        top_neg.update(acc_neg, num_neg)

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Pos_Acc {top_pos.val:.3f} ({top_pos.avg:.3f})\t'
                  'Neg_Acc {top_neg.val:.3f} ({top_neg.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top_pos=top_pos, top_neg=top_neg))

def accuracy(preds, labels):
    preds = preds.view(-1)
    labels = labels.view(-1)
    pos_correct = ((preds >= 0.5) & (labels > 0.5)).sum()
    pos_acc = pos_correct.float()/((labels > 0.5).sum())
    neg_correct = ((preds < 0.5) & (labels < 0.5)).sum()
    neg_acc = neg_correct.float()/((labels < 0.5).sum())
    return pos_acc,(labels > 0.5).sum(), neg_acc, (labels < 0.5).sum()
def adjust_learning_rate(optimizer, epoch):
    global lr
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    new_lr = lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
if __name__ == '__main__':
    main()
