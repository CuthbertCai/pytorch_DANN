"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim

import numpy as np

from models import models
from train import test, train, params
from util import utils
from sklearn.manifold import TSNE

import argparse, sys

import torch
from torch.autograd import Variable


def visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader):
    """
    Evaluate the performance of dann and source only by visualization.

    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :return:
    """
    # Setup the network
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    dataiter = iter(src_test_dataloader)
    s_images, s_labels = dataiter.next()
    s_tags = Variable(torch.zeros((s_labels.size()[0])).type(torch.LongTensor))

    dataiter = iter(tgt_test_dataloader)
    t_images, t_labels = dataiter.next()
    t_tags = Variable(torch.ones((t_labels.size()[0])).type(torch.LongTensor))


    # Compute the embedding of target domain.
    embedding1 = feature_extractor(s_images)
    embedding2 = feature_extractor(t_images)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(), embedding1.detach().numpy())))

    # utils.plot_embedding(source_only_tsne, combined_test_labels.argmax(1), combined_test_domain.argmax(1), 'Source only')
    utils.plot_embedding(dann_tsne, np.concatenate((s_labels.numpy(), t_labels.numpy())),
                         np.concatenate((s_tags.numpy(), t_tags.numpy())), 'Domain Adaptation')



def main(args):
    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader('MNIST')
    src_test_dataloader = utils.get_test_loader('MNIST')
    tgt_train_dataloader = utils.get_train_loader('MNIST_M')
    tgt_test_dataloader = utils.get_test_loader('MNIST_M')

    if args.plot:
        print('Images from training on source domain:')
        utils.displayImages(src_train_dataloader)

        print('Images from test on target domain:')
        utils.displayImages(tgt_test_dataloader)

    # init models
    feature_extractor = models.Extractor()
    class_classifier = models.Class_classifier()
    domain_classifier = models.Domain_classifier()

    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # init criterions
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # init optimizer
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= 0.01, momentum= 0.9)

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
                    src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
        test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader)

    if args.plot:
        visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                             tgt_test_dataloader)



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot', type=bool, default=True, help='plot figures.')

    parser.add_argument('--training_mode', type=str, default='dann', help='which mode to train the model.')


    return parser.parse_args()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
