import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import params

def get_train_loader(dataset):
    """
    Get train dataloader of source domain or target domain
    :return: dataloader
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.MNIST(root= params.source_path, train= True, transform= transform,
                              download= True)

        dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root= params.target_path + '/train', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader

def get_test_loader(dataset):
    """
    Get test dataloader of source domain or target domain
    :return: dataloader
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.MNIST(root= params.source_path, train= False, transform= transform,
                              download= True)

        dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root= params.target_path + '/test', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
    else:
        raise Exception('There is no dataset named {}'.format(str(dataset)))

    return dataloader

def optimizer_scheduler(optimizer, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer