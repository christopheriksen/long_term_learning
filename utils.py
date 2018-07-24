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
import random


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



def load_image_folder(traindir, valdir, normalization=None):

    if normalization != None:

        normalize = transforms.Normalize(mean=normalization[0], std=normalization[1])

        train_dataset = datasets.ImageFolder(traindir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]))

        val_dataset = datasets.ImageFolder(valdir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]))


    # otherwise calculate mean, std to normalize by
    else:

        train_dataset = datasets.ImageFolder(traindir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()      # converts to [0.0, 1.0]
            # normalize
            ]))

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


        train_dataset = datasets.ImageFolder(traindir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]))

        val_dataset = datasets.ImageFolder(valdir, train=True, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]))

    return train_dataset, val_dataset



# def load_rgbd_by_instance(traindir, valdir, normalization=None):
#     #

