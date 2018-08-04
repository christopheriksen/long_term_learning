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





def load_instance_subsets_leave_one_out(instances_per_subset, data_dir, normalization):

    if normalization != None:

        normalize = transforms.Normalize(mean=normalization[0], std=normalization[1])

        dataset_total = datasets.ImageFolder(data_dir, transform= transforms.Compose([
            # transforms.Resize(299),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),      # converts to [0.0, 1.0]
            normalize
            ]))





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

                dataset = load_dataset()    #FIXME      configure file structure with all classes for each instance

                if instance_index == left_out_index:
                    test_datasets.append(dataset)
                    test_instance_names.append(instance)
                else:
                    train_datasets_by_instance.append(dataset)
                    train_instance_names.append(instance)


        # shuffle train datasets
        num_train_instances = len(train_datasets_by_instance)
        train_instance_indices = len(range(num_train_instances))
        random.shuffle(train_instance_indices)

        num_subsets = int(math.ceil(num_train_instances/instances_per_subset))
        train_datasets_by_subset = []
        train_instance_names_by_subset = []

        for i in range(num_subsets-1):
            subset_indices = train_instance_indices[i:i+instances_per_subset]
            subset_datasets_by_instance = train_datasets_by_instance[subset_indices]        # FIXME     right syntax?
            subset_dataset = torch.utils.data.ConcatDataset(subset_datasets_by_instance)
            train_datasets_by_subset.append(subset_dataset)

            subset_instance_names = train_instance_names[subset_indices]                    # FIXME     right syntax?
            train_instance_names_by_subset.append(subset_instance_names)

        subset_indices = train_instance_indices[(num_subsets-1)*instances_per_subset:]
        subset_datasets_by_instance = train_datasets_by_instance[subset_indices]        # FIXME     right syntax?
        subset_dataset = torch.utils.data.ConcatDataset(subset_datasets_by_instance)
        train_datasets_by_subset.append(subset_dataset)

        subset_instance_names = train_instance_names[subset_indices]                    # FIXME     right syntax?
        train_instance_names_by_subset.append(subset_instance_names)


        # concat test datasets
        test_dataset = torch.utils.data.ConcatDataset(test_datasets)


    return train_datasets_by_subset, test_dataset, train_instance_names_by_subset, test_instance_names




    classes = os.listdir(data_dir)
    num_classes = len(classes)

    # for calculating pixel mean
    train_data_all = []

    # group data by object instance
    train_data_instances = []
    train_instance_labels = []
    train_instance_names = []

    test_data_instances = []
    test_instance_labels = []
    test_instance_names= []
    for i in range(num_classes):
        curr_class = classes[i]
        instances = os.listdir(data_dir + curr_class)
        num_instances = len(instances)

        # test instance to leave out of training
        left_out_index = random.randint(0, num_instances-1)

        for instance_index in range(num_instances):
            instance = instances[instance_index]
            instance_examples = []
            examples = os.listdir(data_dir + curr_class + "/" + instance)
            for example in examples:
                img_file = data_dir + curr_class + "/" + instance + "/" + example
                img = cv2.imread(img_file)
                img = resize_and_center_crop(img, 32)   # convert to (32, 32, 3)
                instance_examples.append(img)
            instance_examples = np.array(instance_examples)
            instance_examples = instance_examples/np.float32(255)
            instance_examples = instance_examples.reshape((instance_examples.shape[0], 32, 32, 3)).transpose(0,3,1,2)

            if instance_index == left_out_index:
                test_data_instances.append(instance_examples)
                test_instance_labels.append(i)
                test_instance_names.append(instance)
            else:
                train_data_instances.append(instance_examples)
                train_instance_labels.append(i)
                train_instance_names.append(instance)

                for instance_example in instance_examples:
                    train_data_all.append(instance_example)

    train_data_all = np.array(train_data_all)
    train_data_instances = np.array(train_data_instances)
    test_data_instances = np.array(test_data_instances)


    # subtract pixel mean
    pixel_mean = np.mean(train_data_all, axis=0)

    # subtract pixel mean from train examples
    for instance in train_data_instances:
        instance -= pixel_mean
    # subtract pixel mean from test examples
    for instance in test_data_instances:
        instance -= pixel_mean


    # create test set
    X_test = []
    Y_test = []
    for i in range(test_data_instances.shape[0]):
        y = test_instance_labels[i]
        for x in test_data_instances[i]:
            X_test.append(x)
            Y_test.append(y)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    

    # shuffle instances
    num_train_instances = train_data_instances.shape[0]
    train_instance_indices = list(range(num_train_instances))
    random.shuffle(train_instance_indices)
    X_train_subsets = []
    Y_train_subsets = []
    train_instance_names_by_subset = []


    # create data subsets
    num_subsets = int(math.ceil(num_train_instances/instances_per_subset))
    for i in range(num_subsets-1):
        subset_indices = train_instance_indices[i:i+instances_per_subset]
        subset_xs = []
        subset_ys = []
        subset_instance_names = []
        for index in subset_indices:
            y = train_instance_labels[index]
            subset_instance_names.append(train_instance_names[index])
            for x in train_data_instances[index]:
                subset_xs.append(x)
                subset_ys.append(y)
        # # shuffle current subset
        # num_subset_examples = len(subset_xs)
        # subset_indices = list(range(num_subset_examples))
        # random.shuffle(subset_indices)
        # shuffled_subset_xs = []
        # shuffled_subset_ys = []
        # for index in subset_indices:
        #     shuffled_subset_xs.append(np.array(subset_xs[index]))
        #     shuffled_subset_ys.append(np.array(subset_ys[index]))
        X_train_subsets.append(np.array(subset_xs))
        Y_train_subsets.append(np.array(subset_ys))
        train_instance_names_by_subset.append(subset_instance_names)
        
    subset_indices = train_instance_indices[(num_subsets-1)*instances_per_subset:]
    subset_xs = []
    subset_ys = []
    subset_instance_names = []
    for index in subset_indices:
        y = train_instance_labels[index]
        subset_instance_names.append(train_instance_names[index])
        for x in train_data_instances[index]:
            subset_xs.append(x)
            subset_ys.append(y)
    # # shuffle current subset
    # num_subset_examples = len(subset_xs)
    # subset_indices = list(range(num_subset_examples))
    # random.shuffle(subset_indices)
    # shuffled_subset_xs = []
    # shuffled_subset_ys = []
    # for index in subset_indices:
    #     shuffled_subset_xs.append(np.array(subset_xs[index]))
    #     shuffled_subset_ys.append(np.array(subset_ys[index]))
    X_train_subsets.append(np.array(subset_xs))
    Y_train_subsets.append(np.array(subset_ys))
    train_instance_names_by_subset.append(subset_instance_names)

    for i in range(len(X_train_subsets)):
        X_train_subsets[i] = lasagne.utils.floatX(X_train_subsets[i])
        Y_train_subsets[i] = Y_train_subsets[i].astype('int32')
        # print (X_train_subsets[i].shape)

    return dict(
        # X_train_subsets = lasagne.utils.floatX(X_train_subsets),
        # Y_train_subsets = Y_train_subsets.astype('int32'),
        X_train_subsets = X_train_subsets,
        Y_train_subsets = Y_train_subsets,
        train_instance_names_by_subset = train_instance_names_by_subset,
        # X_valid = lasagne.utils.floatX(X_valid),
        # Y_valid = Y_valid.astype('int32'),
        X_test  = lasagne.utils.floatX(X_test),
        Y_test  = Y_test.astype('int32'),
        test_instance_names = test_instance_names,)







# def load_rgbd_by_instance(traindir, valdir, normalization=None):
#     #

