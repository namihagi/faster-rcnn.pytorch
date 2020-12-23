# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import, division, print_function

import pdb

import numpy as np
import numpy.random as npr
# from scipy.misc import imread
from imageio import imread
from model.utils.blob import im_list_to_blob_PIL, prep_im_for_blob_PIL
from model.utils.config import cfg
from PIL import Image, ImageOps


def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)

    blobs = {'data': im_blob}

    assert len(im_scales) == 1, "Single batch only"
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(
            roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob[0].size[1], im_blob[0].size[0], im_scales[0]]],
        dtype=np.float32)

    blobs['img_id'] = roidb[0]['img_id']

    return blobs


def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # import cv2
        # im = cv2.imread(roidb[i]['image'])
        # im = imread(roidb[i]['image'])
        im = Image.open(roidb[i]['image'])

        if im.mode != "RGB":
            im = im.convert("RGB")

        if roidb[i]['flipped']:
            im = ImageOps.mirror(im)

        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob_PIL(im, target_size)
        im_scales.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob_PIL(processed_ims)

    return blob, im_scales
