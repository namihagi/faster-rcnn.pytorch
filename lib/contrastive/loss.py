import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg

from contrastive.cosine_similarity import (cosine_similarity_for_all_pair,
                                           cosine_similarity_for_grad_stop)


class ContrastiveLossForRoIWithGradStop(nn.Module):
    def __init__(self, iou_threshold=0.8):
        super(ContrastiveLossForRoIWithGradStop, self).__init__()

        self.iou_threshold = iou_threshold
        if cfg.CUDA:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def forward(self,
                feat_a_p, feat_a_z,
                feat_b_p, feat_b_z, iou):
        batch_size = feat_a_p.size(0)

        # leave only the max iou for each box
        iou_max_mask = torch.zeros_like(iou)
        iou_max_mask = iou_max_mask.scatter(
            dim=-1, index=iou.argmax(dim=-1, keepdim=True), value=1
        )
        iou_max_only = torch.mul(iou, iou_max_mask).to(self.device)

        # leave only the iou over threshold
        iou_match = iou_max_only.where(
            iou_max_only >= self.iou_threshold,
            torch.zeros_like(iou_max_only)
        )

        # get indices of matched pairs
        match_idx = iou_match.nonzero(as_tuple=False)
        matched_box_num = match_idx.size(0)

        # get matched features
        # shape of matched_feat_(a|b): (num_matched_box, feat_dim)
        matched_feat_a_p = feat_a_p[match_idx[:, 0], match_idx[:, 1]]
        matched_feat_a_z = feat_a_z[match_idx[:, 0], match_idx[:, 1]]
        matched_feat_b_p = feat_b_p[match_idx[:, 0], match_idx[:, 2]]
        matched_feat_b_z = feat_b_z[match_idx[:, 0], match_idx[:, 2]]

        # negative cosine similarity
        loss_a = cosine_similarity_for_grad_stop(matched_feat_a_p,
                                                 matched_feat_b_z)
        loss_b = cosine_similarity_for_grad_stop(matched_feat_b_p,
                                                 matched_feat_a_z)
        loss = loss_a / 2 + loss_b / 2

        # leave only the iou over threshold
        matched_box_num = torch.where(
            iou_max_only >= self.iou_threshold,
            torch.ones_like(iou_max_only),
            torch.zeros_like(iou_max_only)
        )
        matched_box_num = matched_box_num.sum(dim=[1, 2])
        assert matched_box_num.size(0) == batch_size

        return loss, matched_box_num


class ContrastiveLossForRoI(nn.Module):
    def __init__(self, iou_threshold=0.8, temperature=0.1):
        super(ContrastiveLossForRoI, self).__init__()

        self.iou_threshold = iou_threshold
        self.temp = temperature
        if cfg.CUDA:
            self.device = "cuda"
        else:
            self.device = "cpu"

    def forward(self, feat_a, feat_b, sim, iou):
        """
        args:
            feat_a: (batch_size, max_box_num, feature_dim) tensor of features extracted from the augmented image A.
            feat_b: (batch_size, max_box_num, feature_dim) tensor of features extracted from the augmented image B.
            sim: (batch_size, max_box_num, max_box_num) cosine similarity matrix.
            iou: (batch_size, max_box_num, max_box_num) iou matrix between proposal boxes A and B.
        returns:
            loss: a scalar of torch.float32
        """
        batch_size = feat_a.size(0)
        dim = feat_a.size(-1)

        iou_max_mask = torch.zeros_like(iou)
        iou_max_mask = iou_max_mask.scatter(
            dim=-1, index=iou.argmax(dim=-1, keepdim=True), value=1
        )

        iou_max_only = torch.mul(iou, iou_max_mask).to(self.device)
        iou_match = iou_max_only.where(
            iou_max_only >= self.iou_threshold,
            torch.zeros_like(iou_max_only)
        )

        loss = torch.zeros(batch_size, dtype=torch.float32).to(self.device)
        matched_box_num = torch.zeros_like(loss).long().to(self.device)

        neg_base = torch.arange(batch_size)
        for b_idx in range(batch_size):
            pos_idx = iou_match[b_idx].nonzero(as_tuple=False)
            pos_sim = sim[b_idx, pos_idx[:, 0], pos_idx[:, 1]]
            # pos_sim shape: (matched_box_num)
            match_feat = feat_a[b_idx, pos_idx[:, 1]]
            # feat_to_compare shape: (matched_box_num, feature_dim)

            neg_indices = torch.cat([neg_base[0:b_idx], neg_base[b_idx + 1:]])
            neg_indices = neg_indices.to(self.device)
            neg_feat_a = feat_a.index_select(dim=0, index=neg_indices)
            neg_feat_b = feat_b.index_select(dim=0, index=neg_indices)
            neg_feat = torch.cat([neg_feat_a.view(-1, dim),
                                  neg_feat_b.view(-1, dim)], dim=0)
            # neg_feat shape: ((batch_size - 1) * 2 * matched_box_num, feature_dim)

            neg_sim = cosine_similarity_for_all_pair(match_feat.unsqueeze(0),
                                                     neg_feat.unsqueeze(0))
            # neg_sim shape: (1, matched_box_num, (batch_size - 1) * 2 * matched_box_num)

            # scale by tempereture
            pos_sim = pos_sim / self.temp
            neg_sim = neg_sim / self.temp

            sim_for_loss = torch.cat([pos_sim.unsqueeze(1), neg_sim[0]], dim=1)
            pos_label = torch.zeros(sim_for_loss.size(0)).long()
            loss[b_idx] = F.cross_entropy(sim_for_loss,
                                          pos_label.to(self.device))
            matched_box_num[b_idx] = pos_idx.size(0)

        return loss, matched_box_num
