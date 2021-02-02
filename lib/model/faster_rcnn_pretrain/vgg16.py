# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from model.faster_rcnn_pretrain.head import (prediction_MLP, projection_MLP,
                                             projection_SimCLR)
from model.faster_rcnn_pretrain.pretrained_net import _pretrainedNet
from torch.autograd import Variable


class vgg16(_pretrainedNet):
    def __init__(self, classes, pretrained=False,
                 class_agnostic=False, fix_backbone=True,
                 temperature=0.1, iou_threshold=0.7,
                 grad_stop=False, share_rpn=False, random_rpn=False):

        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic
        self.fix_backbone = fix_backbone

        _pretrainedNet.__init__(self, classes, class_agnostic,
                                temperature, iou_threshold,
                                grad_stop, share_rpn, random_rpn)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({
                k: v for k, v in state_dict.items() if k in vgg.state_dict()
            })

        vgg.classifier = nn.Sequential(
            *list(vgg.classifier._modules.values())[:-1]
        )

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(
            *list(vgg.features._modules.values())[:-1]
        )

        if self.fix_backbone:
            # Fix the layers before conv3:
            for layer in range(10):
                for p in self.RCNN_base[layer].parameters():
                    p.requires_grad = False

        self.RCNN_top = vgg.classifier

        fdim = 4096
        hidden_dim = 1024

        if self.grad_stop:
            self.proj_mlp = projection_MLP(in_dim=fdim,
                                           hidden_dim=fdim,
                                           out_dim=fdim)

            self.pred_mlp = prediction_MLP(in_dim=fdim,
                                           hidden_dim=hidden_dim,
                                           out_dim=fdim)

        else:
            self.proj_mlp = projection_SimCLR(in_dim=fdim,
                                              hidden_dim=fdim,
                                              out_dim=128)

    def projection_head(self, feat):
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.proj_mlp(feat)
        return feat

    def _head_to_tail(self, pool5):
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)
        return fc7
