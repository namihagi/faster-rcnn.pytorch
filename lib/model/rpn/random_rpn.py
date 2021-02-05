import torch
from model.roi_layers import nms
from torch import nn


class RandomRoi(nn.Module):
    def __init__(self,
                 nms_threshold=0.7,
                 num_of_first_box=2000,
                 min_scale_rate=0.1,
                 min_num_of_final_box=5,
                 max_num_of_final_box=50):
        super(RandomRoi, self).__init__()

        self.nms_threshold = nms_threshold
        self.num_of_first_box = num_of_first_box
        self.min_scale_rate = min_scale_rate
        self.min_num_of_final_box = min_num_of_final_box
        self.max_num_of_final_box = max_num_of_final_box

    def forward(self, batch_size, im_info):
        """
        args:
            batch_size: a scalar
            im_info: [height, width, scale] x batch_size
        """
        device = im_info.device

        batch_boxes_init = torch.rand(batch_size, self.num_of_first_box, 4).to(device)
        batch_boxes = torch.zeros_like(batch_boxes_init)

        # batch_boxes[:, :, 0] (x1) < batch_boxes[:, :, 2] (x2)
        batch_boxes[:, :, 0] = torch.where(batch_boxes_init[:, :, 0] < batch_boxes_init[:, :, 2],
                                           batch_boxes_init[:, :, 0], batch_boxes_init[:, :, 2])
        batch_boxes[:, :, 2] = torch.where(batch_boxes_init[:, :, 0] < batch_boxes_init[:, :, 2],
                                           batch_boxes_init[:, :, 2], batch_boxes_init[:, :, 0])

        # batch_boxes[:, :, 1] (y1) < batch_boxes[:, :, 3] (y2)
        batch_boxes[:, :, 1] = torch.where(batch_boxes_init[:, :, 1] < batch_boxes_init[:, :, 3],
                                           batch_boxes_init[:, :, 1], batch_boxes_init[:, :, 3])
        batch_boxes[:, :, 3] = torch.where(batch_boxes_init[:, :, 1] < batch_boxes_init[:, :, 3],
                                           batch_boxes_init[:, :, 3], batch_boxes_init[:, :, 1])

        # pseudo batch_scores
        batch_scores = torch.rand(batch_size, self.num_of_first_box, 1).to(device)
        sorted_batch_scores, order = torch.sort(batch_scores, 1, True)

        num_of_final_box = torch.randint(self.min_num_of_final_box,
                                         self.max_num_of_final_box,
                                         size=(batch_size, )).to(device)
        sum_of_boxes_num = num_of_final_box.sum()
        output_boxes = torch.zeros((sum_of_boxes_num, 5)).to(device)

        s_idx = 0
        for i in range(batch_size):
            shape = im_info[i][:2]
            boxes = batch_boxes[i]
            scores = batch_scores[i]

            # rescale at range
            boxes[:, 0::2] *= shape[1]  # width
            boxes[:, 1::2] *= shape[0]  # height

            # get min length of box
            min_height = shape[0] * self.min_scale_rate
            min_width = shape[1] * self.min_scale_rate

            # calculate box height and width
            box_height = boxes[:, 3] - boxes[:, 1]
            box_width = boxes[:, 2] - boxes[:, 0]

            # get indices larger than threshold
            box_keep_idx = (box_height > min_height) * (box_width > min_width)

            # get boxes and scores to keep
            boxes_keep = boxes[box_keep_idx]
            scores_keep = scores[box_keep_idx]

            # sort boxes and scores about scores
            sorted_scores, order = torch.sort(scores_keep, 0, True)
            sorted_boxes = boxes_keep[order]

            # nms
            keep_idx_i = nms(sorted_boxes.view(-1, 4),
                             sorted_scores.view(-1), self.nms_threshold)
            keep_idx_i = keep_idx_i.long().view(-1)

            # take topN from keep_idx
            keep_idx_i = keep_idx_i[:num_of_final_box[i]]

            # get the final selected boxes
            selected_boxes = sorted_boxes[keep_idx_i]
            # selected_scores = sorted_scores[keep_idx_i]

            e_idx = s_idx + num_of_final_box[i]
            output_boxes[s_idx:e_idx, 0] = i
            output_boxes[s_idx:e_idx, 1:] = selected_boxes.view(-1, 4)
            s_idx = e_idx

        return output_boxes, num_of_final_box


if __name__ == "__main__":
    roi = RandomRoi()
    im_info = torch.tensor(
        [[600.0000, 901.0000, 1.8018],
         [600.0000, 901.0000, 1.8018],
         [600.0000, 901.0000, 1.8018]])
    boxes = roi(3, im_info)
    print(boxes.shape)
