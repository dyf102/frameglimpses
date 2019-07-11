#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision.models.vgg import VGG, make_layers, cfgs, vgg16

class Vgg16(VGG):
    '''
    Get VGG 16 feature with the diamension(4096, 1)
    '''
    def __init__(self):
        super().__init__(make_layers(cfgs['D']))
        self.classifier = self.classifier[:-1]
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, x):
        # here, implement the forward function, keep the feature maps you like
        # and return them
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x