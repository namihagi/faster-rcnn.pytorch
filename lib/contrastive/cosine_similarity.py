import torch


def cosine_similarity_for_all_pair(a, b, eps=1e-8):
    """
    args:
        a: a tensor (batch_size, box_num1, feature_dim)
        b: a tensor (batch_size, box_num2, feature_dim)
    return:
        cosine similarity matrix: (batch_size, box_num1, box_num2)
    """
    a_base = a.norm(dim=-1, keepdim=True)
    b_base = b.norm(dim=-1, keepdim=True)

    a_norm = a / torch.max(a_base, eps * torch.ones_like(a_base))
    b_norm = b / torch.max(b_base, eps * torch.ones_like(b_base))
    sim_mt = torch.bmm(a_norm, b_norm.transpose(1, 2))
    return sim_mt
