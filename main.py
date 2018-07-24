"""
Main script for models
"""
import torch
import torch.nn as nn
import torch.optim as optim

import models
import params
import train
import test
import utils

# prepare the source data and target data
src_train_dataloader = utils.get_train_loader('MNIST')
src_test_dataloader = utils.get_test_loader('MNIST')
tgt_train_dataloader = utils.get_train_loader('MNIST_M')
tgt_test_dataloader = utils.get_test_loader('MNIST_M')

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
    train.train(feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
                src_train_dataloader, tgt_train_dataloader, optimizer, epoch)
    test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader, tgt_test_dataloader)
