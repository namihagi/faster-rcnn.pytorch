"""
this code is refered to https://github.com/PatrickHua/SimSiam
"""

import torch
import torch.nn.functional as F


def cosine_similarity_for_all_pair(a, b, negative=True):
    """
    args:
        a: a tensor (batch_size, box_num1, feature_dim)
        b: a tensor (batch_size, box_num2, feature_dim)
    return:
        cosine similarity matrix: (batch_size, box_num1, box_num2)
    """
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)

    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    if negative:
        sim_mt = -1 * sim_mt
    return sim_mt


def cosine_similarity_for_grad_stop(p, z):
    """
    args:
        p: a tensor of features (num_of_boxes, feat_dim)
        z: a tensor of features (num_of_boxes, feat_dim)
    return:
        negative cosine similarity between p and z
    """
    z = z.detach()

    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=1)
