import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models

import numpy as np
import utils
import models


def main():

    ############ Modifiable ###################
    data_source_dir = '/media/scatha/Data/lifelong_object_learning/training_data'
    # data_source_dir = '/media/ceriksen/Elements/Data/training_data'

    weights_dir = '/home/scatha/lifelong_object_learning/long_term_learning/weights/'
    # weights_dir = '/home/ceriksen/lifelong_object_learning/long_term_learning/weights/'

    orderings_dir = '/home/scatha/lifelong_object_learning/long_term_learning/orderings/'
    # orderings_dir = '/home/ceriksen/lifelong_object_learning/long_term_learning/orderings/'


    # dataset = "imagenet"
    # dataset = "cifar10"
    # dataset = "cifar100"
    dataset = "rgbd-object"

    num_classes = 51


    arch = 'resnet18'
    # arch = 'resnet34'
    # arch = 'resnet50'
    # arch = 'resnet101'
    # arch = 'resnet152'

    # SGD
    optimizer_method = 'sgd'
    # lr = 2.0
    # lr_dec_factor = 0.2
    # lr_dec_freq = 30
    # momentum = 0.0
    # weight_decay = 0.00001 
    lr = 0.01
    lr_dec_factor = 0.1
    lr_dec_freq = 30
    momentum = 0.9
    weight_decay = 1e-4

    # # Adadelta
    # optimizer_method = 'adadelta'
    # lr=0.001
    # alpha=0.9
    # eps = 1e-08
    # weight_decay=0
    # momentum=0
    # centered=False

    # RMSprop
    # optimizer_method = 'rmsprop'

    # batch_size = 128
    batch_size = 256
    start_epoch = 0
    epochs = 90
    print_freq = 10
    workers = 4
    cudnn_benchmark = True

    load_weights = True
    load_ckpt = False
    imagenet_finetune = False
    imagenet_normalization = True
    freeze_weights = False

    weights_load_name = 'resnet18_rgbd_all_imagenet_lr0.01_e90_v1.pth'

    load_order = True
    subset_instance_order_file = 'instance_order_0.txt'
    test_instances_file = 'test_instances_0.txt'
    ############################################

    ## model

    if imagenet_finetune:
        # torchvision resnet
        if arch == 'resnet18':
            model = models.resnet18(pretrained=True, new_num_classes=num_classes)

    # torchvision resnet
    if arch == 'resnet18':
        model = models.resnet18(num_classes=num_classes)
    if arch == 'resnet34':
        model = models.resnet34(num_classes=num_classes)
    if arch == 'resnet50':
        model = models.resnet50(num_classes=num_classes)
    if arch == 'resnet101':
        model = models.resnet101(num_classes=num_classes)
    if arch == 'resnet152':
        model = models.resnet152(num_classes=num_classes)


    # load saved weights
    if load_weights:
        state_dict = torch.load(weights_dir + weights_load_name)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        state_dict = new_state_dict

        model.load_state_dict(state_dict)

    # resume from checkpoint
    if load_ckpt:
        checkpoint = torch.load(weights_dir+ckpt_save_name)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(weights_dir+ckpt_save_name, checkpoint['epoch']))


    cudnn.benchmark = cudnn_benchmark
    model = torch.nn.DataParallel(model).cuda()



    ## define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # SGD
    if optimizer_method == 'sgd':

        optimizer = torch.optim.SGD(model.parameters(), lr,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    # # Adadelta
    # if optimizer_method == 'adadelta':
    #     optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), 
    #                                 lr=lr,
    #                                 rho=rho,
    #                                 eps = eps,
    #                                 weight_decay=weight_decay)

    # # RMSprop
    # if optimizer_method == 'rmsprop':
    # optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), 
    #                             lr=lr,
    #                             alpha=alpha,
    #                             eps = eps,
    #                             weight_decay=weight_decay,
    #                             momentum=momentum,
    #                             centered=centered)



    ## Data loading code
    if dataset == "cifar100":

        traindir = data_source_dir+"/cifar100"
        valdir = data_source_dir+"/cifar100"

        if imagenet_normalization:
            train_dataset, val_dataset = utils.load_CIFAR100(traindir, valdir, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])      # ImageNet pretrain

        else:
            # train_dataset, val_dataset = utils.load_CIFAR100(traindir, valdir, None)
            train_dataset, val_dataset = utils.load_CIFAR100(traindir, valdir, [[0.50704312, 0.48651126, 0.44088557], [1.0, 1.0, 1.0]])
            # train_dataset, val_dataset = utils.load_CIFAR100(traindir, valdir, [[0.50704312, 0.48651126, 0.44088557], [0.26177242, 0.25081211, 0.27087295]])



    if dataset == "imagenet":

        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))


    if dataset == "rgbd-object":

        data_dir = data_source_dir+'/rgbd-dataset/'

        if imagenet_normalization:
            normalization_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]      # ImageNet pretrain

        else:
            # normalization_params = None
            # normalization_params = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
            normalization_params = [[0.52728295, 0.498189, 0.48457545], [1.0, 1.0, 1.0]]
            # normalization_params = [[0.52728295, 0.498189, 0.48457545], [0.17303562, 0.18130174, 0.20389825]]

        if load_order:
            train_dataset, val_dataset = utils.load_rgbd_batch_from_order_file(orderings_dir+subset_instance_order_file, orderings_dir+test_instances_file, data_dir, normalization_params)
        
        else:
            train_dataset = utils.load_rgbd_batch(data_dir, normalization_params)
            val_dataset = train_dataset



    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)



    model.eval()
    train_accuracy = validate(train_loader, model, criterion, print_freq)
    val_accuracy = validate(val_loader, model, criterion, print_freq)

    print ("Train accuracy: " + str(train_accuracy.data.cpu().numpy()))
    print ("Val accuracy: " + str(val_accuracy.data.cpu().numpy()))




def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output, features = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
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
