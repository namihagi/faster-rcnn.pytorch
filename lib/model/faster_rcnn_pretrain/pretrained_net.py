import pdb
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from contrastive import ContrastiveLossForRoI, cosine_similarity_for_all_pair
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.bbox_transform import bbox_overlaps_batch_for_contrastive
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from model.utils.net_utils import (_affine_grid_gen, _affine_theta,
                                   _crop_pool_layer, _smooth_l1_loss)
from torch import dtype
from torch.autograd import Variable

# from model.roi_pooling.modules.roi_pool import _RoIPooling
# from model.roi_align.modules.roi_align import RoIAlignAvg


class _pretrainedNet(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic,
                 temperature=0.1, iou_threshold=0.7):
        super(_pretrainedNet, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.iou_threth = iou_threshold
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, use_rpn_train=False)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                                     1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                                       1.0 / 16.0, 0)

        self.contrastive_loss_fn = ContrastiveLossForRoI(iou_threshold)

    def forward(self, im_aug_1, im_aug_2, im_info, gt_boxes, num_boxes):
        batch_size = im_aug_1.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        # base_feat shape: [batch_size, dim, h, w]
        base_feat_aug_1 = self.RCNN_base(im_aug_1)
        base_feat_aug_2 = self.RCNN_base(im_aug_2)

        # feed base feature map tp RPN to obtain rois
        # roi shape: [batch_size, RPN_POST_NMS_TOP_N, 5]
        rois_aug_1, _, _ = self.RCNN_rpn(base_feat_aug_1, im_info,
                                         gt_boxes, num_boxes)
        rois_aug_2, _, _ = self.RCNN_rpn(base_feat_aug_2, im_info,
                                         gt_boxes, num_boxes)

        rois_aug_1 = Variable(rois_aug_1)
        rois_aug_2 = Variable(rois_aug_2)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'align':
            pooled_feat_1 = self.RCNN_roi_align(base_feat_aug_1,
                                                rois_aug_1.view(-1, 5))
            pooled_feat_2 = self.RCNN_roi_align(base_feat_aug_2,
                                                rois_aug_2.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_1 = self.RCNN_roi_pool(base_feat_aug_1,
                                               rois_aug_1.view(-1, 5))
            pooled_feat_2 = self.RCNN_roi_pool(base_feat_aug_2,
                                               rois_aug_2.view(-1, 5))

        # projection head for contrastive learning
        # for aug_1
        z_feat_1 = self.projection_head(pooled_feat_1)
        p_feat_1 = self.pred_mlp(z_feat_1)

        # reshape (batch_size, max_box_num, out_dim)
        out_dim = z_feat_1.size(-1)
        z_feat_1 = z_feat_1.view(batch_size, -1, out_dim)
        out_dim = p_feat_1.size(-1)
        p_feat_1 = p_feat_1.view(batch_size, -1, out_dim)

        # for aug_2
        z_feat_2 = self.projection_head(pooled_feat_2)
        p_feat_2 = self.pred_mlp(z_feat_2)

        # reshape (batch_size, max_box_num, out_dim)
        out_dim = z_feat_2.size(-1)
        z_feat_2 = z_feat_2.view(batch_size, -1, out_dim)
        out_dim = p_feat_2.size(-1)
        p_feat_2 = p_feat_2.view(batch_size, -1, out_dim)

        # calculate iou
        iou = bbox_overlaps_batch_for_contrastive(rois_aug_1, rois_aug_2)

        # calculate cosine similarity
        loss_1, match_num_1 = self.contrastive_loss_fn(p_feat_1, z_feat_1,
                                                       p_feat_2, z_feat_2, iou)
        loss_2, match_num_2 = self.contrastive_loss_fn(p_feat_2, z_feat_2,
                                                       p_feat_1, z_feat_1,
                                                       iou.transpose(1, 2))

        loss = loss_1 + loss_2
        match_num = torch.cat([match_num_1, match_num_2])

        return loss, match_num

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
