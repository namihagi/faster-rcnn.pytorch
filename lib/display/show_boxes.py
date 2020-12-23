import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from model.utils.config import cfg


def draw_roi(rgb_img, pred, save_path):
    color = 'red'

    plt.figure()
    # plot the image for matplotlib
    # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    plt.imshow(rgb_img)
    plt.savefig(save_path)
    currentAxis = plt.gca()

    detections = pred.cpu().numpy()
    box_num = None
    for i, pt in enumerate(detections):
        if all(pt[1:] == 0):
            box_num = i
            break

        pt = list(pt)[1:]
        coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False,
                                            edgecolor=color, linewidth=2))
    plt.savefig(save_path.replace('.png', '_box.png'))
    plt.close()

    if box_num is None:
        box_num = 2000

    print(f"{os.path.basename(save_path)}: {box_num}")
    return box_num
