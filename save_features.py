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
    lr = 2.0
    lr_dec_factor = 0.2
    lr_dec_freq = 30
    momentum = 0.0
    weight_decay = 0.00001 

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
    start_epoch = 0
    epochs = 70
    print_freq = 10
    workers = 4
    cudnn_benchmark = True

    load_weights = True
    load_ckpt = False
    imagenet_finetune = False
    imagenet_normalization = False
    freeze_weights = False

    weights_load_name = 'resnet18_rgbd_all_no_normalize.pth'
    # weights_save_name = 'resnet18_rgbd_all_no_normalize.pth'
    # ckpt_save_name = 'resnet18_rgbd_all_no_normalize_ckpt.pth'
    # best_ckpt_save_name = 'resnet18_rgbd_all_no_normalize_best_ckpt.pth'

    features_file = '/home/scatha/lifelong_object_learning/long_term_learning/rgbd_features.txt'
    labels_file = '/home/scatha/lifelong_object_learning/long_term_learning/rgbd_labels.txt'
    ############################################

    ## model

    # torchvision resnet
    if arch == 'resnet18':
        model = models.resnet18()
    if arch == 'resnet34':
        model = models.resnet34()
    if arch == 'resnet50':
        model = models.resnet50()
    if arch == 'resnet101':
        model = models.resnet101()
    if arch == 'resnet152':
        model = models.resnet152()



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


    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = cudnn_benchmark



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
            train_dataset  = utils.load_rgbd_batch(data_dir, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])      # ImageNet pretrain

        else:
            # train_dataset = utils.load_rgbd_batch(data_dir, None)
            # train_dataset, val_dataset = utils.load_rgbd_batch(data_dir, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            train_dataset = utils.load_rgbd_batch(data_dir, [[0.52728295, 0.498189, 0.48457545], [1.0, 1.0, 1.0]])
            # train_dataset, val_dataset = utils.load_rgbd_batch(data_dir, [[0.52728295, 0.498189, 0.48457545], [0.17303562, 0.18130174, 0.20389825]])

        val_dataset = train_dataset



    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)





    model.eval()

    num_data_points = 207920
    num_features = 512

    features_file = open(features_filename,'w')
    labels_file = open(labels_filename,'w')
    features_file.write(str(num_data_points) + ' ' + str(num_features) + '\n')

    for i, (input, target) in enumerate(data_loader):

        target = target.cuda(non_blocking=True)

        # compute output
        output, features = model(input)

        target = target.data.cpu().numpy()
        features = features.data.cpu().numpy()

        # num_data_points += 1

        features_file.write(str(features[0][0]))
        for feature in features[0][1:]:
            features_file.write(' ' + str(feature))
            features_file.write('\n')

        labels_file.write(str(target[0]) + '\n')

    features_file.close()
    labels_file.close()

    # print (num_data_points)



if __name__ == '__main__':
    main()
