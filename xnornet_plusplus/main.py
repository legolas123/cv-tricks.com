import argparse
from models.resnet_preact_bin import BinConv2d, resnet18_preact_bin
import os
import shutil
import time
from models import get_model
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#from custom_dataset import RandomDataset
import torch.multiprocessing as mp
import quantization
from utils import create_logger
import logging
import torch.nn.functional as F
from models.resnet_preact_bin import BinConv2d
import torch.optim.lr_scheduler as lr_scheduler

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture:')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num_classes',  default=1000, type=int,
                    help='num_classes')
parser.add_argument('--optimizer',  default="adam", type=str,
                    help='sgd or adam')
parser.add_argument('--model_dir',
                        help='model directory',
                        type=str,
                        default='')
parser.add_argument('--quantize', dest='quantize', action='store_true',
                    help='whether to quantize model', default = False)

scheduler =None
best_prec1 = 0
bin_op = None
tb_writer = None
def main():
    global args, best_prec1, tb_writer
    
    args = parser.parse_args()
    tb_writer = create_logger(args)

    model = get_model(args.arch, pretrained = args.pretrained, num_classes = args.num_classes)

    model = torch.nn.DataParallel(model).cuda()

    # if args.quantize:
    #     global bin_op
    #     bin_op = quantization.Binarize(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    param_grp_1 = []
    for m in model.modules():
        if isinstance(m,BinConv2d):
            param_grp_1.append(m.weight)
            param_grp_1.append(m.alpha)
            param_grp_1.append(m.beta)
            param_grp_1.append(m.gamma)
    param_grp_1_ids = list(map(id, param_grp_1))
    param_grp_2 = list(filter(lambda p: id(p) not in param_grp_1_ids, model.parameters()))

    if args.optimizer == "sgd":
        if args.quantize:
            optimizer = torch.optim.SGD([{'params': param_grp_1, 'weight_decay':0}, 
                {'params': param_grp_2, 'weight_decay':args.weight_decay}],
                lr = args.lr,momentum=args.momentum)
        else:
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        if args.quantize:
            optimizer = torch.optim.Adam([{'params': param_grp_1, 'weight_decay':0}, 
                {'params': param_grp_2, 'weight_decay':args.weight_decay}], 
                    lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    global scheduler
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1, 2)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # Normalize takes first mean and then std
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            #transforms.Resize(256),
            #transforms.RandomCrop(224),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4,
                saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # train_dataset = RandomDataset(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))


    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # val_loader = torch.utils.data.DataLoader(
    #     RandomDataset(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)


    if args.evaluate:
        validate(val_loader, model, criterion, args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    global scheduler
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        #import pdb; pdb.set_trace()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # if args.quantize:
        #     bin_op.binarization()


        # compute output
        output = model(input_var)
        #loss = smooth_loss(output, target_var, 0.1)
        loss = criterion(output[0]["logits"], target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output[0]["logits"].data, target, topk=(1, 5))
        #import pdb; pdb.set_trace()
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # if args.quantize:
        #     bin_op.restore()
        #     bin_op.updateBinaryGradWeight()

        optimizer.step()

        scheduler.step(epoch + i / len(train_loader))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            global_step = epoch + float(i)/len(train_loader)
            tb_writer.add_scalar("train/loss", losses.avg, global_step)
            tb_writer.add_scalar("train/top1", top1.avg, global_step)
            tb_writer.add_scalar("train/top5", top5.avg, global_step)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # if args.quantize:
    #     bin_op.binarization()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output[0]["logits"], target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output[0]["logits"].data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
    # if args.quantize:
    #     bin_op.restore()

    logging.info(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    tb_writer.add_scalar("test/loss", losses.avg, epoch)
    tb_writer.add_scalar("test/top1", top1.avg, epoch)
    tb_writer.add_scalar("test/top5", top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join(args.model_dir, filename)
    torch.save(state, filename)
    if is_best:
        best_file = os.path.join(args.model_dir, 'model_best.pth.tar')
        shutil.copyfile(filename, best_file)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
