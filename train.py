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
import pretrainedmodels



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
    # num_classes = 100


    # arch = 'resnet18'
    # arch = 'resnet34'
    # arch = 'resnet50'
    # arch = 'resnet101'
    # arch = 'resnet152'

    pretrained_model = True
    arch = 'inceptionresnetv2'

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

    batch_size = 128
    # batch_size = 256
    start_epoch = 0
    epochs = 30
    print_freq = 10
    workers = 4
    cudnn_benchmark = True

    load_weights = False
    load_ckpt = False
    imagenet_finetune = True
    imagenet_normalization = True
    freeze_weights = False

    weights_load_name = 'example_load.pth'
    weights_save_name = 'inceptionresnetv2_rgbd_all_imagenet_sgd_lr0.01_e90_v1.pth'
    ckpt_save_name = 'inceptionresnetv2_rgbd_all_imagenet_sgd_lr0.01_e90_v1_ckpt.pth'
    best_ckpt_save_name = 'inceptionresnetv2_rgbd_all_imagenet_sgd_lr0.01_e90_v1_best_ckpt.pth'

    load_order = True
    subset_instance_order_file = 'instance_order_0.txt'
    test_instances_file = 'test_instances_0.txt'

    accuracies_file = '/home/scatha/lifelong_object_learning/long_term_learning/accuracies/inceptionresnetv2_rgbd_all_imagenet_sgd_lr0.01_e90_v1.txt'
    ############################################

    ## model

    # torchvision resnet
    if arch == 'resnet18':
        if imagenet_finetune:
            model = models.resnet18(pretrained=True, new_num_classes=num_classes)
        else:
            model = models.resnet18(num_classes=num_classes)
    # if arch == 'resnet34':
    #     model = models.resnet34(num_classes=num_classes)
    # if arch == 'resnet50':
    #     model = models.resnet50(num_classes=num_classes)
    # if arch == 'resnet101':
    #     model = models.resnet101(num_classes=num_classes)
    # if arch == 'resnet152':
    #     model = models.resnet152(num_classes=num_classes)

    if pretrained_model:
        if arch == 'inceptionresnetv2':
            img_size = 299
            if imagenet_finetune:
                model = pretrainedmodels.__dict__['inceptionresnetv2']()
                if freeze_weights:
                    for param in model.parameters():
                        param.requires_grad = False
                model.last_linear = nn.Linear(1536, num_classes)
                # for param in model.last_linear.parameters():
                #     param.requires_grad = True
            else:
                model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=num_classes, pretrained=False)


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

    if optimizer_method == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr = 0.001,
                                    alpha=0.9,
                                    eps = 1e-08,
                                    weight_decay=0,
                                    momentum=0,
                                    centered=False)



    ## Data loading code
    if dataset == "cifar100":

        traindir = data_source_dir+"/cifar100"
        valdir = data_source_dir+"/cifar100"

        if pretrained_model:
            if imagenet_normalization:
                normalization_params = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]      # ImageNet pretrain pretrainedmodels
            else:
                # normalization_params = None
                normalization_params = [[0.50704312, 0.48651126, 0.44088557], [1.0, 1.0, 1.0]]
                # normalization_params = [[0.50704312, 0.48651126, 0.44088557], [0.26177242, 0.25081211, 0.27087295]]

        else:
            img_size = 224
            if imagenet_normalization:
                normalization_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]      # ImageNet pretrain torchvision
            else:
                # normalization_params = None
                normalization_params = [[0.50704312, 0.48651126, 0.44088557], [1.0, 1.0, 1.0]]
                # normalization_params = [[0.50704312, 0.48651126, 0.44088557], [0.26177242, 0.25081211, 0.27087295]]

        train_dataset, val_dataset = utils.load_CIFAR100(traindir, valdir, img_size, normalization_params)



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

        if pretrained_model:
            img_size = 299
            if imagenet_normalization:
                normalization_params = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]      # ImageNet pretrain pretrainedmodels
            else:
                # normalization_params = None
                # normalization_params = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                normalization_params = [[0.52728295, 0.498189, 0.48457545], [1.0, 1.0, 1.0]]
                # normalization_params = [[0.52728295, 0.498189, 0.48457545], [0.17303562, 0.18130174, 0.20389825]]

        else:
            img_size = 224
            if imagenet_normalization:
                normalization_params = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]      # ImageNet pretrain torchvision
            else:
                # normalization_params = None
                # normalization_params = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
                normalization_params = [[0.52728295, 0.498189, 0.48457545], [1.0, 1.0, 1.0]]
                # normalization_params = [[0.52728295, 0.498189, 0.48457545], [0.17303562, 0.18130174, 0.20389825]]

        if load_order:
            train_dataset, val_dataset = utils.load_rgbd_batch_from_order_file(orderings_dir+subset_instance_order_file, orderings_dir+test_instances_file, data_dir, img_size, normalization_params)
        
        else:
            train_dataset = utils.load_rgbd_batch(data_dir, img_size, normalization_params)
            val_dataset = train_dataset



    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)



    ## Train

    # best_prec1 = validate(val_loader, model, criterion, print_freq)
    # early_stopping_buffer = []
    # early_stopping_buffer.append(best_prec1)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(start_epoch, epochs):

        start_time = time.time()

        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_freq)

        # evaluate on validation set
        # prec1 = validate(val_loader, model, criterion, print_freq)
        train_accuracy = validate(train_loader, model, criterion, print_freq).data.cpu().numpy()
        val_accuracy = validate(val_loader, model, criterion, print_freq).data.cpu().numpy()

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)


        # # early stopping
        # if early_stopping:
        #     if len(early_stopping_buffer) == (patience+1):

        #         better = True
        #         for prec in early_stopping_buffer:
        #             if prec1 < prec:
        #                 better = False
        #         if better:
        #             print ("Stopping")
        #             break

        #         early_stopping_buffer.pop(0)
        #         early_stopping_buffer.append(prec1)

        #     else:
        #         early_stopping_buffer.append(prec1)

        # # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best, weights_dir + ckpt_save_name, weights_dir + best_ckpt_save_name)

        print ("Epoch time: " + str(time.time() - start_time))

    # save model
    print ("Saving model")
    torch.save(model.state_dict(), weights_dir + weights_save_name)


    # best_checkpoint = torch.load(weights_dir + best_ckpt_save_name)
    # model.load_state_dict(best_checkpoint['state_dict'])

    # validate(val_loader, model, criterion, print_freq)

    # save accuracies
    f = open(accuracies_file, "w")

    for accuracy in train_accuracies:
        f.write(str(accuracy))
        f.write ('\n')
    f.write ('\n')
    for accuracy in val_accuracies:
        f.write(str(accuracy))
        f.write ('\n')
    f.write('\n')



    # best_checkpoint = torch.load('model_best.pth.tar')
    # model.load_state_dict(best_checkpoint['state_dict'])
    # torch.save(model.state_dict(), weights_dir + ckpt_save_name)



def train(train_loader, model, criterion, optimizer, epoch, print_freq):
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

        target = target.cuda(non_blocking=True)

        # compute output
        output, features = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))


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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename = 'model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


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


def adjust_learning_rate(optimizer, epoch, lr, lr_dec_factor=0.1, lr_dec_freq=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (lr_dec_factor ** (epoch // lr_dec_freq))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
