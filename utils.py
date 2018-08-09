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
import math
import re


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

        train_dataset = datasets.ImageFolder(traindir, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]))

        val_dataset = datasets.ImageFolder(valdir, transform= transforms.Compose([
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

        train_dataset = datasets.ImageFolder(traindir, transform= transforms.Compose([
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

        print ("Train mean: " + str(data_mean))
        print ("Train std (approx): " + str(data_std))

        normalize = transforms.Normalize(mean=data_mean.tolist(), std = [1.0, 1.0, 1.0])


        train_dataset = datasets.ImageFolder(traindir, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]))

        val_dataset = datasets.ImageFolder(valdir, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ]))

    return train_dataset, val_dataset





def load_rgbd_instance_subsets_leave_one_out(instances_per_subset, data_dir, normalization):

    if normalization != None:

        normalize = transforms.Normalize(mean=normalization[0], std=normalization[1])

        # dataset_total = datasets.ImageFolder(data_dir, transform= transforms.Compose([
        #     # transforms.Resize(299),
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
        #     # transforms.RandomResizedCrop(224),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),      # converts to [0.0, 1.0]
        #     normalize
        #     ]))


        # dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)


        # for i, (input, target) in enumerate(dataloader):
        #     # numpy_image = data['image'].numpy()
        #     numpy_image = input.numpy()
        #     batch_mean = np.mean(numpy_image, axis=(0,2,3))
        #     batch_std = np.std(numpy_image, axis=(0,2,3))
        #     data_mean.append(batch_mean)
        #     data_std.append(batch_std)
        # data_mean = np.array(data_mean).mean(axis=0, dtype=np.float32)
        # data_std = np.array(data_std).mean(axis=0, dtype=np.float32)    # approx, not true std


        train_datasets_by_instance = []
        test_datasets = []
        train_instance_names = []
        test_instance_names = []

        classes = os.listdir(data_dir)
        num_classes = len(classes)

        # load datasets by instance
        for i in range(num_classes):
            curr_class = classes[i]
            instances = os.listdir(data_dir + curr_class)
            num_instances = len(instances)

            # test instance to leave out of training
            left_out_index = random.randint(0, num_instances-1)

            for instance_index in range(num_instances):
                instance = instances[instance_index]

                dataset = datasets.ImageFolder(data_dir + curr_class + '/' + instance, transform= transforms.Compose([
                    # transforms.Resize(299),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),      # converts to [0.0, 1.0]
                    normalize
                    ]))

                if instance_index == left_out_index:
                    test_datasets.append(dataset)
                    test_instance_names.append(instance)
                else:
                    train_datasets_by_instance.append(dataset)
                    train_instance_names.append(instance)


        # shuffle train datasets
        num_train_instances = len(train_datasets_by_instance)
        train_instance_indices = list(range(num_train_instances))
        random.shuffle(train_instance_indices)

        num_subsets = int(math.ceil(num_train_instances/instances_per_subset))
        train_datasets_by_subset = []
        train_instance_names_by_subset = []

        for i in range(num_subsets-1):
            subset_indices = train_instance_indices[i:i+instances_per_subset]
            subset_datasets_by_instance = []
            subset_instance_names = []
            for subset_index in subset_indices:
                subset_datasets_by_instance.append(train_datasets_by_instance[subset_index])
                subset_instance_names.append(train_instance_names[subset_index])

            subset_dataset = torch.utils.data.dataset.ConcatDataset(subset_datasets_by_instance)
            train_datasets_by_subset.append(subset_dataset)
            train_instance_names_by_subset.append(subset_instance_names)

        # last iteration
        subset_indices = train_instance_indices[(num_subsets-1)*instances_per_subset:]
        subset_datasets_by_instance = []
        subset_instance_names = []
        for subset_index in subset_indices:
            subset_datasets_by_instance.append(train_datasets_by_instance[subset_index])
            subset_instance_names.append(train_instance_names[subset_index])

        subset_dataset = torch.utils.data.dataset.ConcatDataset(subset_datasets_by_instance)
        train_datasets_by_subset.append(subset_dataset)
        train_instance_names_by_subset.append(subset_instance_names)


        # concat test datasets
        test_dataset = torch.utils.data.dataset.ConcatDataset(test_datasets)


    return train_datasets_by_subset, test_dataset, train_instance_names_by_subset, test_instance_names





def load_instance_subsets_from_order_file(train_list, test_list, data_dir, normalization):

    train_instance_names_by_subset = []
    train_lines = [line.rstrip('\n') for line in open(train_list)]
    for line in train_lines:
        train_instance_names_by_subset.append(line.split())

    test_instance_names = []
    test_lines = [line.rstrip('\n') for line in open(test_list)]
    test_instance_names = test_lines[0].split()


    if normalization != None:

        normalize = transforms.Normalize(mean=normalization[0], std=normalization[1])

        # train datasets
        train_datasets_by_subset = []
        for instance_names in train_instance_names_by_subset:

            subset_datasets = []
            for instance_name in instance_names:
                class_name = re.match("^.*_[0-9]*$", instance_name)
                print (class_name)

                dataset = datasets.ImageFolder(data_dir + class_name + '/' + instance_name, transform= transforms.Compose([
                    # transforms.Resize(299),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),      # converts to [0.0, 1.0]
                    normalize
                    ]))
                subset_datasets.append(dataset)
            subset_dataset = torch.utils.data.dataset.ConcatDataset(subset_datasets)
            train_datasets_by_subset.append(subset_dataset)


        # test dataset
        test_datasets = []
        for instance_name in test_instance_names:
            class_name = re.match("^.*_[0-9]*$", instance_name)
            print (class_name)

            dataset = datasets.ImageFolder(data_dir + class_name + '/' + instance_name, transform= transforms.Compose([
                # transforms.Resize(299),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),      # converts to [0.0, 1.0]
                normalize
                ]))
            test_datasets.append(dataset)
        test_dataset = torch.utils.data.dataset.ConcatDataset(test_datasets)


    return train_datasets_by_subset, test_dataset






def load_rgbd_batch(data_dir, normalization):

    if normalization == None:

        instance_datasets = []
        classes = os.listdir(data_dir)
        num_classes = len(classes)

        # load datasets by instance
        for i in range(num_classes):
            curr_class = classes[i]
            instances = os.listdir(data_dir + curr_class)
            num_instances = len(instances)

            for instance_index in range(num_instances):
                instance = instances[instance_index]
                dataset = datasets.ImageFolder(data_dir + curr_class + '/' + instance, transform= transforms.Compose([
                    # transforms.Resize(299),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()      # converts to [0.0, 1.0]
                    # normalize
                    ]))
                instance_datasets.append(dataset)

        # concat test datasets
        dataset = torch.utils.data.dataset.ConcatDataset(instance_datasets)


        # calculate mean to normalize by
        data_mean = []
        data_std = []
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
        for i, (input, target) in enumerate(dataloader):
            # numpy_image = data['image'].numpy()
            numpy_image = input.numpy()
            batch_mean = np.mean(numpy_image, axis=(0,2,3))
            batch_std = np.std(numpy_image, axis=(0,2,3))
            data_mean.append(batch_mean)
            data_std.append(batch_std)
        data_mean = np.array(data_mean).mean(axis=0, dtype=np.float32)
        data_std = np.array(data_std).mean(axis=0, dtype=np.float32)    # approx, not true std

        print (data_mean)
        print (data_std)

        normalize = transforms.Normalize(mean=data_mean.tolist(), std = [1.0, 1.0, 1.0])



        instance_datasets = []

        # load datasets by instance
        for i in range(num_classes):
            curr_class = classes[i]
            instances = os.listdir(data_dir + curr_class)
            num_instances = len(instances)

            for instance_index in range(num_instances):
                instance = instances[instance_index]
                dataset = datasets.ImageFolder(data_dir + curr_class + '/' + instance, transform= transforms.Compose([
                    # transforms.Resize(299),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),      # converts to [0.0, 1.0]
                    normalize
                    ]))
                instance_datasets.append(dataset)

        # concat test datasets
        normalized_dataset = torch.utils.data.dataset.ConcatDataset(instance_datasets)


    else:

        normalize = transforms.Normalize(mean=normalization[0], std=normalization[1])
        instance_datasets = []
        classes = os.listdir(data_dir)
        num_classes = len(classes)

        # load datasets by instance
        for i in range(num_classes):
            curr_class = classes[i]
            instances = os.listdir(data_dir + curr_class)
            num_instances = len(instances)

            for instance_index in range(num_instances):
                instance = instances[instance_index]
                dataset = datasets.ImageFolder(data_dir + curr_class + '/' + instance, transform= transforms.Compose([
                    # transforms.Resize(299),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    # transforms.RandomResizedCrop(224),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),      # converts to [0.0, 1.0]
                    normalize
                    ]))
                instance_datasets.append(dataset)

        # concat test datasets
        normalized_dataset = torch.utils.data.dataset.ConcatDataset(instance_datasets)


    return normalized_dataset