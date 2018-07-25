"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim

from models import models
from train import test, train, params
from util import utils

import argparse, sys



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



def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot', type=bool, default=True, help='plot figures.')
    parser.add_argument('--training_mode', type=str, default='dann', help='which mode to train the model.')


    return parser.parse_args()



if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
