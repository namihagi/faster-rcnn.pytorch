import torch
import torch.nn as nn
from model.utils.config import cfg

from contrastive.cosine_similarity import cosine_similarity_for_grad_stop


class ContrastiveLossForRoI(nn.Module):
    def __init__(self, iou_threshold=0.8):
        super(ContrastiveLossForRoI, self).__init__()

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
