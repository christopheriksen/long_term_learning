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
import random
import utils
import models
import math
from sklearn.metrics.pairwise import pairwise_distances

from PIL import Image
from random import shuffle
import kmedoids
import csv
import sys
from sklearn.decomposition import PCA


def main(selection_method, distillation, use_ewc, ewc_lambda, weights_save_name, accuracies_file, normalize_features, weight_features, train_batch, use_dim_red, num_dim_red_components, order_number, dictionary_size):

    ############ Modifiable ###################
    data_source_dir = '/media/scatha/Data1/lifelong_object_learning/training_data'
    # data_source_dir = '/media/ceriksen/Elements/Data/training_data'

    weights_dir = '/home/scatha/lifelong_object_learning/long_term_learning/weights/'
    # weights_dir = '/home/ceriksen/lifelong_object_learning/long_term_learning/weights/'

    orderings_dir = '/home/scatha/lifelong_object_learning/long_term_learning/orderings/'
    # orderings_dir = '/home/ceriksen/lifelong_object_learning/long_term_learning/orderings/'


    # dataset = 'imagenet'
    # dataset = 'cifar10'
    # dataset = 'cifar100'
    dataset = 'rgbd-object'

    num_classes = 51
    # num_classes = 100


    arch = 'resnet18'
    # arch = 'resnet34'
    # arch = 'resnet50'
    # arch = 'resnet101'
    # arch = 'resnet152'

    pretrained_model = False
    # arch = 'inceptionresnetv2'

    # SGD
    optimizer_method = 'sgd'
    # lr = 2.0
    # lr_dec_factor = 0.2
    # lr_dec_freq = 30
    # momentum = 0.0
    # weight_decay = 0.00001 

    # lr = 0.01
    # momentum = 0.9
    # weight_decay = 1e-4
    lr_dec_factor = 0.1
    lr_dec_freq = 30

    lr = 1e-2
    momentum = 0.0
    weight_decay = 0.0

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

    batch_size = 32
    start_epoch = 0
    # epochs = 70
    epochs = 10
    print_freq = 10
    workers = 4
    cudnn_benchmark = True

    load_weights = False
    load_ckpt = False
    imagenet_finetune = True
    imagenet_normalization = True
    freeze_weights = False

    # distillation = True
    distillation_merged = False
    # use_ewc = False
    # ewc_mode = 'class'
    # ewc_mode = 'dataset'
    # ewc_mode = 'consolidated'
    # ewc_lambda = 100.0

    num_subsets = 10
    instances_per_subset = 10
    # dictionary_size = 2550
    # dictionary_size = 5000
    num_exemplars_per_class = int(dictionary_size/num_classes)
    # normalize_features = True

    # selection_method = 'random'
    dist_metric = 'sqeuclidean'

    weights_load_name = 'example_load.pth'
    # weights_save_name = 'resnet18_imagenet_cifar100_iter_no_coreset_subsetsize_10_sgd_lr_1e-2_e10_b_32_4.pth'
    # weights_save_name = 'resnet18_imagenet_cifar100_iter_random_distil_subsetsize_10_dic_10_sgd_lr_1e-2_e10_b_32_0.pth'
    # weights_save_name = 'resnet18_imagenet_cifar100_iter_ewc_lambda_100_sgd_lr_1e-2_e10_b_32_0.pth'
    # weights_save_name_base = 'resnet18_imagenet_cifar100_mean_approx_norm_sgd_1e-3_b256__50imgs_0_'
    ckpt_save_name = 'ckpt.pth'
    best_ckpt_save_name = 'model_best.pth.tar'

    load_order = True
    # subset_instance_order_file = 'cifar100_instance_order_' + str(order_number) + '.txt'
    subset_instance_order_file = 'instance_order_' + str(order_number) + '.txt'
    test_instances_file = 'test_instances_' + str(order_number) + '.txt'

    # accuracies_file = '/home/scatha/lifelong_object_learning/long_term_learning/accuracies/cifar100/resnet18_imagenet_cifar100_iter_random_distil_subsetsize_10_dic_10_sgd_lr_1e-2_e10_b_32_0.txt'
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
    distillation_criterion = torch.nn.BCELoss().cuda()

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
    if dataset == 'cifar100':

        traindir = data_source_dir+'/cifar100'
        valdir = data_source_dir+'/cifar100'

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

        cifar_dataset, test_dataset = utils.load_CIFAR100(traindir, valdir, img_size, normalization_params)

        # randomly sample train data points for each subset
        if load_order == False:
            num_data_points = 5000
            random_indices = list(range(num_data_points))
            random.shuffle(random_indices)
            index_groups = []
            subset_size = int(math.floor(num_data_points/num_subsets))

            for i in range(num_subsets-1):
                prev_index = subset_size*i
                next_index = subset_size*(i+1)
                index_groups.append(random_indices[prev_index:next_index])
            prev_index = subset_size*(num_subsets-1)
            index_groups.append(random_indices[prev_index:])

            # write order file
            f = open(orderings_dir + subset_instance_order_file, "w")
            for index_group in index_groups:
                for index in index_group:
                    f.write(str(index))
                    f.write(' ')
                f.write("\n")
            f.close()

        # read in subset index order per subset from file
        else:
            index_groups = []
            train_lines = [line.rstrip('\n') for line in open(orderings_dir + subset_instance_order_file)]
            for line in train_lines:
                index_groups.append(line.split())
            for index_group in index_groups:
                for i in range(len(index_group)):
                    index_group[i] = int(index_group[i])


        train_datasets_by_subset = []
        for index_group in index_groups:
            dataset = torch.utils.data.dataset.Subset(cifar_dataset, index_group)
            train_datasets_by_subset.append(dataset)

        first_train_dataset = train_datasets_by_subset[0]



    if dataset == 'imagenet':

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



    if dataset == 'rgbd-object':

        # data_dir = data_source_dir+'/rgbd-dataset_instances/'
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
                normalization_params = [[0.52728295, 0.498189, 0.48457545], [1.0, 1.0, 1.0]]        # FIXME norm includes test data
                # normalization_params = [[0.52728295, 0.498189, 0.48457545], [0.17303562, 0.18130174, 0.20389825]]


        if load_order:
            train_datasets_by_subset, test_dataset = utils.load_instance_subsets_from_order_file(orderings_dir+subset_instance_order_file, orderings_dir+test_instances_file, data_dir, img_size, normalization_params)

        else:
            train_datasets_by_subset, test_dataset, train_instance_names_by_subset, test_instance_names = utils.load_rgbd_instance_subsets_leave_one_out(instances_per_subset, data_dir, img_size, normalization_params)

            # write instance ordering
            f = open(orderings_dir + subset_instance_order_file, "w")
            for subset_instances in train_instance_names_by_subset:
                for instance in subset_instances:
                    f.write(instance)
                    f.write(' ')
                f.write("\n")
            f.close()

            # write test instance names
            f = open(orderings_dir + test_instances_file, "w")
            for instance_name in test_instance_names:
                f.write(instance_name)
                f.write(' ')
            f.close()

        first_train_dataset = train_datasets_by_subset[0]

        


    # Iterate over data subsets

    num_subsets = len(train_datasets_by_subset)
    print ("Num subsets: " + str(num_subsets))   
    cum_train_dataset = None
    exemplar_dataset = None
    cum_train_accuracies = []
    first_train_accuracies = []
    test_accuracies = []


    if use_ewc:
        fisher = {}
        optpar = {}


    for subset_iter in range(num_subsets):

        print ("Subset iter: " + str(subset_iter))

        train_dataset = train_datasets_by_subset[subset_iter]
        val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        first_train_loader = torch.utils.data.DataLoader(
            first_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)


        if subset_iter == 0:
            cum_train_dataset = train_dataset       # cum dataset for test metrics
            if distillation != True:
                combined_train_dataset = train_dataset
        else:
            cum_train_dataset = torch.utils.data.dataset.ConcatDataset([cum_train_dataset, train_dataset])      # cum dataset for test metrics

            if distillation != True:
                if selection_method != None:
                    # add stored exemplars to training set
                    combined_train_dataset = torch.utils.data.dataset.ConcatDataset([train_dataset, exemplar_dataset])
                else:
                    combined_train_dataset = train_dataset

        cum_train_loader = torch.utils.data.DataLoader(
            cum_train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

        if distillation != True:
            train_loader = torch.utils.data.DataLoader(
                combined_train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True)

        if use_ewc == True:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=workers, pin_memory=True)

        if train_batch:
            # new model
            model = models.resnet18(pretrained=True, new_num_classes=num_classes)
            model = torch.nn.DataParallel(model).cuda()


        if distillation:
            # precompute values for coreset
            old_output = None
            if exemplar_dataset != None:
                model.eval()
                exemplar_dataset_loader = torch.utils.data.DataLoader(
                    exemplar_dataset, batch_size=batch_size, shuffle=False,
                    num_workers=workers, pin_memory=True)

                # old_output = torch.zeros(len(exemplar_dataset), num_classes).cuda()
                old_output = []
                for i, (input, target) in enumerate(exemplar_dataset_loader):
                    target = target.cuda(non_blocking=True)
                    output, features = model(input)
                    softmax_output = torch.nn.functional.sigmoid(output)
                    # old_output[i] = softmax_output.data
                    old_output.append(softmax_output.data)
                # old_output = old_output.cuda(non_blocking=True)


        ## Train

        # best_prec1 = validate(val_loader, model, criterion, print_freq)
        # early_stopping_buffer = []
        # early_stopping_buffer.append(best_prec1)

        for epoch in range(start_epoch, epochs):

            start_time = time.time()

            # adjust_learning_rate(optimizer, epoch, lr)



            if distillation:
                train_distillation(train_dataset, exemplar_dataset, old_output, model, criterion, distillation_criterion, optimizer, batch_size, workers, num_classes, distillation_merged)

            if use_ewc:
                train_ewc(train_loader, model, criterion, optimizer, epoch, print_freq, fisher, optpar, ewc_lambda, subset_iter)

            if train_batch:
                train(cum_train_loader, model, criterion, optimizer, epoch, print_freq, ewc=None)

            # train for one epoch
            if (distillation != True) and (use_ewc != True) and (train_batch != True):
                train(train_loader, model, criterion, optimizer, epoch, print_freq, ewc=None)

            # # evaluate on validation set
            # prec1 = validate(val_loader, model, criterion, print_freq)


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

        validate(val_loader, model, criterion, print_freq)

        # best_checkpoint = torch.load(weights_dir + best_ckpt_save_name)
        # model.load_state_dict(best_checkpoint['state_dict'])



        ## Exemplars 
        model.eval()
        if selection_method != None:  

            if distillation == True:
                if subset_iter != 0:
                    # add stored exemplars to training set
                    combined_train_dataset = torch.utils.data.dataset.ConcatDataset([train_dataset, exemplar_dataset])
                else:
                    combined_train_dataset = train_dataset

            exemplar_pool_loader = torch.utils.data.DataLoader(
                combined_train_dataset, batch_size=1, shuffle=False,
                num_workers=workers, pin_memory=True)


            indices_by_class = [[] for i in range(num_classes)]
            features_by_class = [[] for i in range(num_classes)]

            if use_dim_red:

                # learn pca
                values = []
                for index, (input_img, target) in enumerate(exemplar_pool_loader):
                    output, features = model(input_img)
                    features = features.data.cpu().numpy()[0]
                    if normalize_features:
                        features = features/np.linalg.norm(features)
                    values.append(features)
                values = np.array(values)

                # pca = PCA(n_components=1536)
                pca = PCA(n_components=num_dim_red_components)
                pca.fit(values)

                # group features by class, applying pca
                for index, (input_img, target) in enumerate(exemplar_pool_loader):

                    output, features = model(input_img)

                    target = target.cuda(non_blocking=True)
                    target = target.data.cpu().numpy()[0]
                    features = features.data.cpu().numpy()[0]

                    indices_by_class[target].append(index)

                    if normalize_features:
                        features = features/np.linalg.norm(features)
                    features_by_class[target].append(features)
                    
                for class_index in range(num_classes):
                    indices_by_class[class_index] = np.array(indices_by_class[class_index])
                    features_by_class[class_index] = np.array(features_by_class[class_index])
                    if features_by_class[class_index].shape[0] > 0:
                        features_by_class[class_index] = pca.transform(features_by_class[class_index])

            else:
                for index, (input_img, target) in enumerate(exemplar_pool_loader):

                    output, features = model(input_img)

                    target = target.cuda(non_blocking=True)
                    target = target.data.cpu().numpy()[0]
                    features = features.data.cpu().numpy()[0]

                    indices_by_class[target].append(index)

                    if normalize_features:
                        features = features/np.linalg.norm(features)
                    features_by_class[target].append(features)

                for class_index in range(num_classes):
                    indices_by_class[class_index] = np.array(indices_by_class[class_index])
                    features_by_class[class_index] = np.array(features_by_class[class_index])


            # selection procedure
            exemplar_indices_by_class = [[] for i in range(num_classes)]

            if selection_method == 'random':
                for class_index in range(num_classes):
                    np.random.shuffle(indices_by_class[class_index])
                    exemplar_indices_by_class[class_index] = indices_by_class[class_index][:num_exemplars_per_class]


            if selection_method == 'kmedoids':

                for class_index in range(num_classes):

                    if (indices_by_class[class_index].shape[0] > num_exemplars_per_class):

                        if weight_features:
                            weights = model.module.fc.weight
                            class_weights = weights.data.cpu().numpy()[class_index]
                            class_weights = np.absolute(class_weights)
                            normalized_class_weights = class_weights/np.linalg.norm(class_weights)
                            distance_weights = normalized_class_weights
                        else:
                            distance_weights = None

                        # calculate distance matrix
                        distances = pairwise_distances(features_by_class[class_index], metric=dist_metric, n_jobs=1, w=distance_weights)
                        M, C = kmedoids.kMedoids(distances, num_exemplars_per_class)

                        for index in M:
                            exemplar_indices_by_class[class_index].append(indices_by_class[class_index][index])

                    else:
                        exemplar_indices_by_class[class_index] = indices_by_class[class_index]


            if selection_method == 'mean_approx':

                for class_index in range(num_classes):

                    if (indices_by_class[class_index].shape[0] > num_exemplars_per_class):

                        features_by_class[class_index] = features_by_class[class_index].T
                                      
                        # Herding procedure : ranking of the potential exemplars
                        mu  = np.mean(features_by_class[class_index],axis=1)
                        w_t = mu
                        iter_herding     = 0
                        iter_herding_eff = 0
                        new_prototypes = []
                        while (len(new_prototypes) < min(num_exemplars_per_class, len(indices_by_class[class_index]))):
                            tmp_t   = np.dot(w_t,features_by_class[class_index])
                            ind_max = np.argmax(tmp_t)
                            iter_herding_eff += 1
                            new_prototypes.append(indices_by_class[class_index][ind_max])
                            w_t = w_t+mu-features_by_class[class_index][:,ind_max]

                        exemplar_indices_by_class[class_index] = new_prototypes

            
            # save exemplars as dataset
            exemplar_indices = []
            for class_index in range(num_classes):
                for exemplar_index in exemplar_indices_by_class[class_index]:
                    exemplar_indices.append(exemplar_index)

            # print ("Num exemplar indices: " + str(len(exemplar_indices)))

            exemplar_dataset = torch.utils.data.dataset.Subset(combined_train_dataset, exemplar_indices)
            # print ("Num exemplar dataset: " + str(len(exemplar_dataset)))

        # only performed on new data (not exemplars)


        ## EWC
        if use_ewc:
            fisher[subset_iter] = []
            optpar[subset_iter] = []
            for p in model.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                optpar[subset_iter].append(pd)
                fisher[subset_iter].append(pg)


        # if ewc != None:
        #     if ewc.mode == 'class':
        #         # group train_dataset by class
        #         indices_by_class = [[] for i in range(num_classes)]
        #         new_data_loader = torch.utils.data.DataLoader(
        #             train_dataset, batch_size=1, shuffle=False,
        #             num_workers=workers, pin_memory=True)

        #         model.eval()
        #         for index, (input_img, target) in enumerate(new_data_loader):
        #             target = target.cuda(non_blocking=True)
        #             target = target.data.cpu().numpy()[0]
        #             indices_by_class[target].append(index)

        #         # calculate fisher information by class
        #         for class_index in range(num_classes):
        #             class_dataset = torch.utils.data.dataset.Subset(train_dataset, indices_by_class[class_index])

        #             # calculate fisher information
        #             # ewc.F[class] += new_F         FIXME
        #             # ewc.means[class] = new_mean       FIXME

        #     if ewc.mode == 'dataset':
        #         # calculate fisher information of train_dataset
        #         # ewc.F.insert(new_F, 0)        FIXME
        #         # ewc.means.insert(new_mean, 0)     FIXME
        #         ewc.num_datasets += 1

        #     if ewc.mode == 'consolidated':
        #         # calculate fisher information of train_dataset
        #         # ewc.F += new_F     FIXME
        #         # ewc.mean = new_mean


        ## Distillation



        # validate(val_loader, model, criterion, print_freq)
        cum_train_accuracy = validate(cum_train_loader, model, criterion, print_freq)
        first_train_accuracy = validate(first_train_loader, model, criterion, print_freq)
        test_accuracy = validate(test_loader, model, criterion, print_freq)

        cum_train_accuracies.append(cum_train_accuracy.data.cpu().numpy())
        first_train_accuracies.append(first_train_accuracy.data.cpu().numpy())
        test_accuracies.append(test_accuracy.data.cpu().numpy())


        # # save model
        # print ("Saving model")
        # torch.save(model.state_dict(), weights_dir + weights_save_name_base + str() + '.pth')
        # torch.save(model.state_dict(), weights_dir + weights_save_name)


    # # save model
    # print ("Saving model")
    # torch.save(model.state_dict(), weights_dir + weights_save_name)

    # # write coreset images to file
    # exemplar_pool_loader = torch.utils.data.DataLoader(
    #     exemplar_dataset, batch_size=1, shuffle=False,
    #     num_workers=workers, pin_memory=True)

    # output_by_class = [[] for i in range(num_classes)]
    # for index, (input_img, target) in enumerate(exemplar_pool_loader):

    #     output, features = model(input_img)

    #     target = target.cuda(non_blocking=True)
    #     target = target.data.cpu().numpy()[0]
    #     output = output.data.cpu().numpy()[0]

    #     # un-normalize output

    #     output_by_class[target].append(output)

    # for class_index in range(num_classes):
    #     for i in range(len(output_by_class[class_index])):
    #         output_img_name = output_img_base + str(class_index) + '/' + str(class_index) + '_' + str(i) + '.jpg'


    # test_accuracy = validate(test_loader, model, criterion, print_freq)
    # cum_train_accuracy = validate(cum_train_loader, model, criterion, print_freq)



    # save accuracies
    f = open(accuracies_file, "w")

    for accuracy in cum_train_accuracies:
        f.write(str(accuracy))
        f.write (', ')
    f.write ('\n')
    for accuracy in first_train_accuracies:
        f.write(str(accuracy))
        f.write (', ')
    f.write('\n')
    for accuracy in test_accuracies:
        f.write(str(accuracy))
        f.write (', ')
    f.write('\n')


def train_distillation(train_dataset, coreset, old_output, model, criterion, distillation_criterion, optimizer, batch_size, workers, num_classes, distillation_merged):

    # switch to train mode
    model.train()
    batch_criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    batch_distillation_criterion = torch.nn.BCELoss(size_average=False).cuda()

    if coreset != None:
        num_train_data = len(train_dataset)
        num_coreset_data = len(coreset)
        num_data = num_train_data + num_coreset_data
        # loss = torch.Tensor([0.0]).cuda(non_blocking=True)

        # new train data
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

        # coreset data
        coreset_loader = torch.utils.data.DataLoader(
            coreset, batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(non_blocking=True)
            output, features = model(input)
            # loss += (batch_criterion(output, target)/num_data)
            loss = criterion(output, target)

            if distillation_merged:
                for i_c, (input_c, target_c) in enumerate(coreset_loader):
                    target_c = target_c.cuda(non_blocking=True)
                    output_c, features_c = model(input_c)

                    new_output = torch.nn.functional.sigmoid(output_c)
                    # new_output = new_output.cuda(non_blocking=True).squeeze()

                    # loss += (batch_distillation_criterion(new_output, old_output[i])/num_data).data
                    loss += (batch_distillation_criterion(new_output, old_output[i])/num_coreset_data)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        if distillation_merged != True:
            for i, (input, target) in enumerate(coreset_loader):
                target = target.cuda(non_blocking=True)
                output, features = model(input)

                new_output = torch.nn.functional.sigmoid(output)
                # new_output = new_output.cuda(non_blocking=True).squeeze()

                # loss += (batch_distillation_criterion(new_output, old_output[i])/num_data).data
                loss = distillation_criterion(new_output, old_output[i])

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


    else:
        num_train_data = len(train_dataset)
        # loss = torch.Tensor([0.0]).cuda(non_blocking=True)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

        for i, (input, target) in enumerate(train_loader):
            target = target.cuda(non_blocking=True)
            output, features = model(input)
            # loss += (batch_criterion(output, target)/num_train_data).data
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loss = torch.autograd.Variable(loss, requires_grad = True)
        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()


# def train_distillation(train_dataset, coreset, model, criterion, distillation_criteron, optimizer, batch_size, workers, num_classes):

#     # switch to train mode
#     model.train()

#     batch_criterion = nn.CrossEntropyLoss(weight=None, size_average=False).cuda()

#     if coreset != None:
#         num_new_data = len(train_dataset)
#         num_coreset = len(coreset)
#         total_num = num_new_data + num_coreset
#         combined_train_dataset = torch.utils.data.dataset.ConcatDataset([coreset, train_dataset])

#         # combined_train_loader = torch.utils.data.DataLoader(
#         #     combined_train_dataset, batch_size=batch_size, shuffle=True,
#         #     num_workers=workers, pin_memory=True)

#         # precompute values for coreset
#         # model.eval()
#         old_output = torch.zeros(num_coreset, num_classes).cuda()
#         for index in list(range(num_coreset)):
#             (input, target) = combined_train_dataset[index]
#             input = input.cuda(non_blocking=True)
#             input = input.unsqueeze(0)
#             # index = index.cuda(non_blocking=True)
#             output, features = model(input)
#             softmax_output = torch.nn.functional.sigmoid(output)
#             old_output[index] = softmax_output.data
#         old_output = old_output.cuda(non_blocking=True)

#         # iterate over data
#         # model.train()
#         batch_indices = list(torch.utils.data.sampler.BatchSampler(torch.utils.data.sampler.RandomSampler(range(total_num)), batch_size=batch_size, drop_last=False))
#         for batch in batch_indices:

#             # print ("single batch")
#             # loss = torch.Tensor([0.0]).cuda(non_blocking=True)
#             first = True
#             batch_loader = torch.utils.data.DataLoader(
#                 batch_subset, batch_size=1, shuffle=False,
#                 num_workers=workers, pin_memory=True)
#             for i, (input, target) in enumerate(batch_loader):
#                 index = batch[i]
#                 target = target.cuda(non_blocking=True)
#                 output, features = model(input)

#                 # # new data
#                 # if index >= num_coreset:
#                 #     # instance_loss = criterion(output, target)
#                 #     loss += criterion(output, target)
#                 #     # print ("ce")
#                 #     # print (loss.shape)
#                 #     # print (loss)

#                 # # distillation loss for coreset
#                 # else:
#                 #     # instance_loss = torch.nn.BCELoss(F.sigmoid(output), old_output[index])
#                 #     new_output = torch.nn.functional.sigmoid(output).data
#                 #     new_output = new_output.cuda(non_blocking=True).squeeze()
#                 #     loss = distillation_criteron(new_output, old_output[index])
#                 #     # print ("bce")
#                 #     # print (loss.shape)
#                 #     # print (loss)

#                 if first:
#                     loss  = criterion(output, target)
#                 else:
#                     loss  += criterion(output, target)

#                 first = False

#             loss = loss/len(batch)


#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


#     else:
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=batch_size, shuffle=True,
#             num_workers=workers, pin_memory=True)

#         for i, (input, target) in enumerate(train_loader):
#             target = target.cuda(non_blocking=True)
#             output, features = model(input)
#             loss = criterion(output, target)

#             # compute gradient and do SGD step
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()


def train_ewc(train_loader, model, criterion, optimizer, epoch, print_freq, fisher, optpar, ewc_lambda, num_datasets_seen):
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

        # ewc regularization
        for dataset in range(num_datasets_seen):
            for i, p in enumerate(model.parameters()):
                l = ewc_lambda * torch.autograd.Variable(fisher[dataset][i])
                l = l * (p - torch.autograd.Variable(optpar[dataset][i])).pow(2)
                loss += l.sum()


        # # if using elastic weight consolidation, modify loss
        # if ewc != None:
        #     if ewc.mode == 'class':
        #         for class_index in range(ewc.num_classes):
        #             # loss += (ewc.importance/ewc.num_classes) * ewc.F[class_index]     # FIXME
        #     if ewc.mode == 'dataset':
        #         for dataset_index in range(ewc.num_datasets):
        #             # loss += (ewc.discount^dataset_index)(ewc.importance/ewc.num_datasets) * ewc.F[dataset_index]      # FIXME
        #     if ewc.mode == 'consolidated':
        #         # loss += ewc.importance * ewc.F      # FIXME

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


def train(train_loader, model, criterion, optimizer, epoch, print_freq, ewc=None):
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


        # # if using elastic weight consolidation, modify loss
        # if ewc != None:
        #     if ewc.mode == 'class':
        #         for class_index in range(ewc.num_classes):
        #             # loss += (ewc.importance/ewc.num_classes) * ewc.F[class_index]     # FIXME
        #     if ewc.mode == 'dataset':
        #         for dataset_index in range(ewc.num_datasets):
        #             # loss += (ewc.discount^dataset_index)(ewc.importance/ewc.num_datasets) * ewc.F[dataset_index]      # FIXME
        #     if ewc.mode == 'consolidated':
        #         # loss += ewc.importance * ewc.F      # FIXME

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
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))

    selection_method = sys.argv[1]
    distillation = bool(int(sys.argv[2]))
    use_ewc = bool(int(sys.argv[3]))
    ewc_lambda = float(sys.argv[4])
    weights_save_name = sys.argv[5]
    accuracies_file = sys.argv[6]
    normalize_features = bool(int(sys.argv[7]))
    weight_features = bool(int(sys.argv[8]))
    train_batch = bool(int(sys.argv[9]))
    use_dim_red = bool(int(sys.argv[10]))
    num_dim_red_components = int(sys.argv[11])
    order_number = int(sys.argv[12])
    dictionary_size = int(sys.argv[13])
    main(selection_method, distillation, use_ewc, ewc_lambda, weights_save_name, accuracies_file, normalize_features, weight_features, train_batch, use_dim_red, num_dim_red_components, order_number, dictionary_size)
