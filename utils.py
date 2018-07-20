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
import torchvision.models as models

import numpy as np


def load_CIFAR100(traindir, valdir, normalization=None):

    if normalization != None:

        normalize = transforms.Normalize(mean=normalization[0], std=normalization[1])

        train_dataset = datasets.CIFAR100(traindir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]),
            download=False)

        val_dataset = datasets.CIFAR100(valdir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]),
            download=False)


    # otherwise calculate mean, std to normalize by
    else:

        train_dataset = datasets.CIFAR100(traindir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()      # converts to [0.0, 1.0]
            # normalize
            ]),
            download=False)

        # subtract out mean
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)
        data_mean = []
        data_std = []
        # for i, data in enumerate(dataloader, 0):
        for i, (input, target) in enumerate(dataloader):
            # numpy_image = data['image'].numpy()
            numpy_image = input.numpy()
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std = np.std(numpy_image, axis=(0,2,3))
            data_mean.append(batch_mean)
            data_std.append(batch_std)
        data_mean = np.array(data_mean).mean(axis=0, dtype=np.float32)
        data_std = np.array(data_std).mean(axis=0, dtype=np.float32)    # approx, not true std

        print ("Train mean: str(data_mean)")
        print ("Train std (approx): str(data_std)")

        normalize = transforms.Normalize(mean=data_mean.tolist(), std = [1.0, 1.0, 1.0])


        train_dataset = datasets.CIFAR100(traindir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]),
            download=False)

        val_dataset = datasets.CIFAR100(valdir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]),
            download=False)

    return train_dataset, val_dataset






# def load_CIFAR100_random_subsets(num_subsets, data_path):
    
#     x = np.concatenate(xs)/np.float32(255)
#     y = np.concatenate(ys)
#     x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
#     x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)
#     # subtract per-pixel mean
#     pixel_mean = np.mean(x[0:50000],axis=0)
#     x -= pixel_mean



#     # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                          std=[0.229, 0.224, 0.225])

#     train_dataset = datasets.CIFAR100(traindir, train=True, transform= transforms.Compose([
#         transforms.Resize(299),
#         # transforms.RandomResizedCrop(299),
#         # transforms.RandomHorizontalFlip(),
#         transforms.ToTensor()
#         # normalize
#         ]),
#         download=True)

#     xs = []
#     ys = []
#     for (x,y) in train_dataset:
#         xs.append(x)
#         ys.append(y)

#     xs = np.array(xs)
#     xs = xs/np.float32(255) # scale between 0 and 1

#     train_dataset[,0] = train_dataset/np.float32(255)
#     # subtract per-pixel mean
#     pixel_mean = np.mean(xs,axis=0)
#     xs -= pixel_mean



#     # shuffle training data
#     num_data_points = 50000
#     indices = np.array(list(range(num_data_points)))
#     np.random.shuffle(indices)
#     X_train = []
#     Y_train = []

#     for index in indices:
#         X_train.append(x[index])
#         Y_train.append(y[index])

#     # split training data into subsets
#     X_train_subsets = []
#     Y_train_subsets = []

#     subset_size = int(math.floor(num_data_points/num_subsets))

#     for i in range(num_subsets-1):
#         prev_index = subset_size*i
#         next_index = subset_size*(i+1)
#         X_train_subsets.append(X_train[prev_index:next_index])
#         Y_train_subsets.append(Y_train[prev_index:next_index])

#     prev_index = subset_size*(num_subsets-1)
#     X_train_subsets.append(X_train[prev_index:])
#     Y_train_subsets.append(Y_train[prev_index:])

#     X_train_subsets = np.array(X_train_subsets)
#     Y_train_subsets = np.array(Y_train_subsets)