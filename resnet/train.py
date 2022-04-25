import os
import sys
import shutil
import numpy as np
import time, datetime
import torch
import random
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed

sys.path.append("../")
from utils.utils import *
from utils import KD_loss
from torchvision import datasets, transforms
from torch.autograd import Variable
from resnet import resnet18, resnet34, resnet50
import torchvision.models as models

parser = argparse.ArgumentParser("n2uq")
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet101', help='teacher model')
parser.add_argument('--student', type=str, default='resnet18', help='student model')
parser.add_argument('--n_bit', type=int, default=2, help='number of bits')
parser.add_argument('--quantize_downsample', type=str, default='True', help='quantize downsampling layer or not')
parser.add_argument('-j', '--workers', default=40, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
args = parser.parse_args()

resnet_dict = {'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50}

real_model_dict_rewrite_1x1 = {'resnet18': 'real_res18.pth.tar',
        'resnet34': 'real_res34.pth.tar',
        'resnet50': 'real_res50.pth.tar'}

real_model_dict = {'resnet18': 'resnet18-f37072fd.pth',
        'resnet34': 'resnet34-b627a593.pth',
        'resnet50': 'resnet50-0676ba61.pth'}


CLASSES = 1000

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        sys.exit(1)
    start_t = time.time()

    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)

    # load model
    model_teacher = models.__dict__[args.teacher](pretrained=True)
    model_teacher = nn.DataParallel(model_teacher).cuda()
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()

    if args.quantize_downsample == 'True' or args.quantize_downsample ==  '1':
        args.quantize_downsample = True
    else:
        args.quantize_downsample = False

    model_student = resnet_dict[args.student](args.n_bit, args.quantize_downsample)
    logging.info('student:')
    logging.info(model_student)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    criterion_kd = KD_loss.DistributionLoss()

    all_parameters = model_student.parameters()
    weight_parameters = []
    alpha_parameters = []

    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 and 'bias' not in pname:
            print('weight_param:', pname)
            weight_parameters.append(p)
        if 'quan1.a' in pname or 'quan2.a' in pname or 'scale' in pname or 'start' in pname:
            print('alpha_param:', pname)
            alpha_parameters.append(p)

    weight_parameters_id = list(map(id, weight_parameters))
    alpha_parameters_id = list(map(id, alpha_parameters))
    other_parameters1 = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    other_parameters = list(filter(lambda p: id(p) not in alpha_parameters_id, other_parameters1))

    optimizer = torch.optim.Adam(
            [{'params' : alpha_parameters, 'lr': args.learning_rate / 10},
            {'params' : other_parameters, 'lr': args.learning_rate},
            {'params' : weight_parameters, 'weight_decay': args.weight_decay, 'lr': args.learning_rate}],
            betas=(0.9,0.999))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)
    start_epoch = 0
    best_top1_acc= 0

    if args.quantize_downsample:
        checkpoint_tar = os.path.join(args.save, real_model_dict_rewrite_1x1[args.student])
    else:
        checkpoint_tar = os.path.join(args.save, real_model_dict[args.student])

    if not os.path.exists(checkpoint_tar):
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        import gdown # pip install gdown
        if args.student == 'resnet18':
            if args.quantize_downsample:
                model_id = '1BfWhr1hzaS_5zHpQDREVRiPOCmXEWC7q'
            else:
                model_id = '1DhnwgWsAYOTIrTJLaMkT7FQD7oHXMz41'
        elif args.student == 'resnet34':
            if args.quantize_downsample:
                model_id = '1kvrloDtpJjhgyF4bX0z0CvHFeCKOHj_B'
            else:
                model_id = '1kbQfXr4YS9XsboYV_-nG4kVHe9MVk5N1'
        elif args.student == 'resnet50':
            if args.quantize_downsample:
                model_id = '1cySVFj5PV0ngJlvLmNRcIufzOLRF_ut0'
            else:
                model_id = '1IVYC2ngycFWAZlUtOPAdahZbzAwgX6VU'

        gdown.download(id=model_id, output=checkpoint_tar, quiet=False)

    checkpoint = torch.load(checkpoint_tar)
    model_student.load_state_dict(checkpoint, strict=False)
    model_student = nn.DataParallel(model_student).cuda()

    checkpoint_tar = os.path.join(args.save, args.student + '_' + str(args.n_bit) + 'bit_quantize_downsample_' + str(args.quantize_downsample), 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar)
        start_epoch = checkpoint['epoch'] + 1
        best_top1_acc = checkpoint['best_top1_acc']
        model_student.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    # adjust the learning rate according to the checkpoint
    for epoch in range(start_epoch):
        scheduler.step()

    # load training data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # data augmentation
    crop_scale = 0.08
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    train_dataset = datasets.ImageFolder(
        traindir,
        transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load validation data
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # train the model
    epoch = start_epoch

    while epoch < args.epochs:
        train_obj, train_top1_acc,  train_top5_acc = train(epoch,  train_loader, model_student, model_teacher, criterion_kd, optimizer, scheduler)
        valid_obj, valid_top1_acc, valid_top5_acc = validate(epoch, val_loader, model_student, criterion, args)

        is_best = False
        if valid_top1_acc > best_top1_acc:
            best_top1_acc = valid_top1_acc
            is_best = True

        save_checkpoint({
            'epoch': epoch,
            'state_dict': model_student.state_dict(),
            'best_top1_acc': best_top1_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, os.path.join(args.save, args.student + '_' + str(args.n_bit) + 'bit_quantize_downsample_' + str(args.quantize_downsample)))

        epoch += 1

    training_time = (time.time() - start_t) / 36000
    print('total training time = {} hours'.format(training_time))


def train(epoch, train_loader, model_student, model_teacher, criterion, optimizer, scheduler):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model_student.train()
    model_teacher.eval()
    end = time.time()
    scheduler.step()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        images = images.cuda()
        target = target.cuda()

        # compute outputy
        logits_student = model_student(images)
        logits_teacher = model_teacher(images)
        loss = criterion(logits_student, logits_teacher)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)   #accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    return losses.avg, top1.avg, top5.avg

def validate(epoch, val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            progress.display(i)

        print(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


if __name__ == '__main__':
    main()
