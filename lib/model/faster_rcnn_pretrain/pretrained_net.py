import sys

import torch
import torch.nn as nn
from contrastive import ContrastiveLossForRoI, cosine_similarity_for_all_pair
from contrastive.cosine_similarity import cosine_similarity_for_grad_stop
from contrastive.loss import ContrastiveLossForRoIWithGradStop
from model.roi_layers import ROIAlign, ROIPool
from model.rpn.bbox_transform import bbox_overlaps_batch_for_contrastive
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.rpn.random_rpn import RandomRoi
from model.rpn.rpn import _RPN
from model.utils.config import cfg
from torch.autograd import Variable


class _pretrainedNet(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic,
                 temperature=0.1, iou_threshold=0.7,
                 grad_stop=False, share_rpn=False,
                 random_rpn=False):
        super(_pretrainedNet, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        self.iou_threth = iou_threshold
        self.grad_stop = grad_stop
        self.share_rpn = share_rpn
        self.random_rpn = random_rpn

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model, use_rpn_train=self.random_rpn)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RandRoI = RandomRoi()

        self.RCNN_roi_pool = ROIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                                     1.0 / 16.0)
        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                                       1.0 / 16.0, 0)

        if not self.share_rpn:
            if self.grad_stop:
                self.contrastive_loss_fn = \
                    ContrastiveLossForRoIWithGradStop(iou_threshold)
            else:
                self.contrastive_loss_fn = \
                    ContrastiveLossForRoI(iou_threshold, temperature)

    def forward(self, im_aug_1, im_aug_2, im_info, gt_boxes, num_boxes):
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        batch_size = im_aug_1.size(0)

        # feed image data to base model to obtain base feature map
        # base_feat shape: [batch_size, dim, h, w]
        base_feat_aug_1 = self.RCNN_base(im_aug_1)
        base_feat_aug_2 = self.RCNN_base(im_aug_2)

        # feed base feature map tp RPN to obtain rois
        # roi shape: [batch_size, RPN_POST_NMS_TOP_N, 5]
        if self.random_rpn:
            rois_aug_1, list_of_box_num = self.RandRoI(batch_size, im_info)
            psuedo_boxes = torch.zeros_like(gt_boxes)
            s_idx = 0
            for i in range(batch_size):
                num_psuedo = list_of_box_num[i]
                psuedo_boxes[i, :num_psuedo, 4] = 1
                e_idx = list_of_box_num[i] + s_idx
                psuedo_boxes[i, :num_psuedo, :4] = \
                    rois_aug_1[s_idx:e_idx, 1:]
            _, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat_aug_1, im_info,
                                                           psuedo_boxes, list_of_box_num)
        else:
            rois_aug_1, _, _ = self.RCNN_rpn(base_feat_aug_1, im_info,
                                             gt_boxes, num_boxes)
            if not self.share_rpn:
                rois_aug_2, _, _ = self.RCNN_rpn(base_feat_aug_2, im_info,
                                                 gt_boxes, num_boxes)

        rois_aug_1 = Variable(rois_aug_1)
        if self.share_rpn or self.random_rpn:
            rois_aug_2 = rois_aug_1
        else:
            rois_aug_2 = Variable(rois_aug_2)
        # do roi pooling based on predicted rois

        # pooling by the same roi
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

        # feed pooled features to top model
        # pooled feat shape: [batch_size * RPN_POST_NMS_TOP_N, out_dim]
        pooled_feat_1 = self._head_to_tail(pooled_feat_1)
        pooled_feat_2 = self._head_to_tail(pooled_feat_2)

        # projection head for contrastive learning
        # for aug_1
        z_feat_1 = self.proj_mlp(pooled_feat_1)
        if self.grad_stop:
            p_feat_1 = self.pred_mlp(z_feat_1)

        # for aug_2
        z_feat_2 = self.proj_mlp(pooled_feat_2)
        if self.grad_stop:
            p_feat_2 = self.pred_mlp(z_feat_2)

        # calculate cosine similarity
        if self.share_rpn:
            loss_1 = cosine_similarity_for_grad_stop(p_feat_1,
                                                     z_feat_2)
            loss_2 = cosine_similarity_for_grad_stop(p_feat_2,
                                                     z_feat_1)

            loss = loss_1 / 2 + loss_2 / 2
            if self.random_rpn:
                return loss, rpn_loss_cls, rpn_loss_bbox
            else:
                return loss

        else:
            # calculate iou
            iou = bbox_overlaps_batch_for_contrastive(rois_aug_1,
                                                      rois_aug_2)

            # reshape (batch_size, max_box_num, out_dim)
            out_dim = z_feat_1.size(-1)
            z_feat_1 = z_feat_1.view(batch_size, -1, out_dim)
            out_dim = z_feat_2.size(-1)
            z_feat_2 = z_feat_2.view(batch_size, -1, out_dim)

            if self.grad_stop:
                # reshape (batch_size, max_box_num, out_dim)
                out_dim = p_feat_1.size(-1)
                p_feat_1 = p_feat_1.view(batch_size, -1, out_dim)
                out_dim = p_feat_2.size(-1)
                p_feat_2 = p_feat_2.view(batch_size, -1, out_dim)

                # calculate cosine similarity
                loss_1, num_of_matched_box_1 = \
                    self.contrastive_loss_fn(p_feat_1, z_feat_1,
                                             p_feat_2, z_feat_2,
                                             iou)
                loss_2, num_of_matched_box_2 = \
                    self.contrastive_loss_fn(p_feat_2, z_feat_2,
                                             p_feat_1, z_feat_1,
                                             iou.transpose(1, 2))

            else:
                # calculate cosine similarity
                cos_sim = cosine_similarity_for_all_pair(z_feat_1, z_feat_2,
                                                         negative=False)

                # calculate loss fot z_feat_1
                loss_1, num_of_matched_box_1 = \
                    self.contrastive_loss_fn(z_feat_1, z_feat_2,
                                             cos_sim, iou)
                # calculate loss fot z_feat_2
                loss_2, num_of_matched_box_2 = \
                    self.contrastive_loss_fn(z_feat_2, z_feat_1,
                                             cos_sim.transpose(1, 2),
                                             iou.transpose(1, 2))

            loss = loss_1 + loss_2
            num_of_matched_box = torch.cat([num_of_matched_box_1,
                                            num_of_matched_box_2])
            return loss, num_of_matched_box

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
