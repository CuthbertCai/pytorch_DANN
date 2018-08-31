import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from train import params
from sklearn.manifold import TSNE


import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import os, time
from data import SynDig


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

        data = datasets.MNIST(root= params.mnist_path, train= True, transform= transform,
                              download= True)

        dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)


    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnistm_path + '/train', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)

    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data1 = datasets.SVHN(root=params.svhn_path, split='train', transform=transform, download=True)
        data2 = datasets.SVHN(root= params.svhn_path, split= 'extra', transform = transform, download= True)

        data = torch.utils.data.ConcatDataset((data1, data2))

        dataloader = DataLoader(dataset=data, batch_size=params.batch_size, shuffle=True)
    elif dataset == 'SynDig':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = SynDig.SynDig(root= params.syndig_path, split= 'train', transform= transform, download= False)

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

        data = datasets.MNIST(root= params.mnist_path, train= False, transform= transform,
                              download= True)

        dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            # transforms.RandomCrop((28)),
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        data = datasets.ImageFolder(root=params.mnistm_path + '/test', transform= transform)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std = params.dataset_std)
        ])

        data = datasets.SVHN(root= params.svhn_path, split= 'test', transform = transform, download= True)

        dataloader = DataLoader(dataset = data, batch_size= params.batch_size, shuffle= True)
    elif dataset == 'SynDig':
        transform = transforms.Compose([
            transforms.CenterCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
        ])

        data = SynDig.SynDig(root= params.syndig_path, split= 'test', transform= transform, download= False)

        dataloader = DataLoader(dataset= data, batch_size= params.batch_size, shuffle= True)
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



def displayImages(dataloader, length=8, imgName=None):
    """
    Randomly sample some images and display
    :param dataloader: maybe trainloader or testloader
    :param length: number of images to be displayed
    :param imgName: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return

    # randomly sample some images.
    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    # process images so they can be displayed.
    images = images[:length]

    images = torchvision.utils.make_grid(images).numpy()
    images = images/2 + 0.5
    images = np.transpose(images, (1, 2, 0))



    if params.fig_mode == 'display':

        plt.imshow(images)
        plt.show()

    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'displayImages' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')


        plt.imsave(imgName, images)
        plt.close()

    # print labels
    print(' '.join('%5s' % labels[j].item() for j in range(length)))



def plot_embedding(X, y, d, title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image

    :return:
    """
    if params.fig_mode is None:
        return


    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i]/1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)

    if params.fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()


    if params.fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        if imgName is None:
            imgName = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        plt.savefig(imgName)
        plt.close()