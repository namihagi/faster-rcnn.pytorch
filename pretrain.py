# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pprint
import sys
import time
import contrastive

import numpy as np
import torch
import torch.nn as nn
from model.faster_rcnn_pretrain.resnet import resnet
from model.faster_rcnn_pretrain.vgg16 import vgg16
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import (adjust_learning_rate, clip_gradient,
                                   save_checkpoint)
from roi_data_layer.roibatchLoader_contrastive import roibatchLoader
from roi_data_layer.roidb import combined_roidb
from torch import optim
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler

import _init_paths


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--pro_name', dest='pro_name',
                        help='project name which identifies model params',
                        type=str)
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--anchor', dest='anchor',
                        help='when dataset is coco, you can use anchor settings for pasval_voc or coco',
                        default=None, type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--iou_threshold', dest='iou_threshold',
                        help='the threshold of iou for contrastive learning',
                        default=0.5, type=float)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--not_fix_backbone', dest='not_fix_backbone',
                        help='whether weights of backbone is fixed',
                        action='store_false')
    parser.add_argument('--without_IN_pretrain',
                        dest='without_IN_pretrain',
                        help='whether backbone weights pretrained by ImageNet is loaded',
                        action='store_false')
    parser.add_argument('--grad_stop', dest='grad_stop',
                        help='whether to use gradient-stop module',
                        action='store_true')
    parser.add_argument('--share_rpn', dest='share_rpn',
                        help='whether to share rpn output',
                        action='store_true')
    parser.add_argument('--random_rpn', dest='random_rpn',
                        help='whether to use random roi instead of RPN',
                        action='store_true')

# config region proposal network
    parser.add_argument('--rpn_top_n', dest='rpn_top_n',
                        help='the number of proposal boxes by RPN',
                        default=500, type=int)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--scheduler', dest='scheduler',
                        help='whether use lr scheduler',
                        action='store_true')

# set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
# log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    args = parser.parse_args()

    # dependency
    if args.random_rpn:
        args.share_rpn = True
    if args.share_rpn:
        args.grad_stop = True
        args.scheduler = True
        args.optimizer = "sgd_decay"

    return args


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size,
                                         train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(
            self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(
            self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat(
                (self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data


if __name__ == '__main__':
    init_time = time.time()

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        if args.anchor == "coco":
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.anchor == "pascal_voc":
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                             'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        else:
            assert False, "you need --anchor option"
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                         'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    if args.large_scale:
        args.cfg_file = "cfgs/{}_ls.yml".format(args.net)
    else:
        args.cfg_file = "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # for pretrain, change RPN_POST_NMS_TOP_N
    cfg.TRAIN.RPN_POST_NMS_TOP_N = args.rpn_top_n

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = False
    cfg.USE_GPU_NMS = args.cuda
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    iters_per_epoch = int(train_size / args.batch_size)

    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = os.path.join(
        args.save_dir, args.pro_name, args.net, args.dataset
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sampler_batch = sampler(train_size, args.batch_size)

    use_bgr = args.net in ["vgg16", "res50", "res101"]
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,
                             imdb.num_classes, training=True, use_bgr=use_bgr,
                             flip=True, is_augmented=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             sampler=sampler_batch, num_workers=args.num_workers)

    # initilize the tensor holder here.
    im_data_aug_1 = torch.FloatTensor(1)
    im_data_aug_2 = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data_aug_1 = im_data_aug_1.cuda()
        im_data_aug_2 = im_data_aug_2.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data_aug_1 = Variable(im_data_aug_1)
    im_data_aug_2 = Variable(im_data_aug_2)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    # initilize the network here.
    fasterRCNN = None
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb.classes,
                           pretrained=args.without_IN_pretrain,
                           class_agnostic=args.class_agnostic,
                           fix_backbone=args.not_fix_backbone,
                           iou_threshold=args.iou_threshold,
                           grad_stop=args.grad_stop,
                           share_rpn=args.share_rpn,
                           random_rpn=args.random_rpn)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb.classes, 101,
                            pretrained=args.without_IN_pretrain,
                            class_agnostic=args.class_agnostic,
                            fix_backbone=args.not_fix_backbone,
                            iou_threshold=args.iou_threshold,
                            grad_stop=args.grad_stop,
                            share_rpn=args.share_rpn,
                            random_rpn=args.random_rpn)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb.classes, 50,
                            pretrained=args.without_IN_pretrain,
                            class_agnostic=args.class_agnostic,
                            fix_backbone=args.not_fix_backbone,
                            iou_threshold=args.iou_threshold,
                            grad_stop=args.grad_stop,
                            share_rpn=args.share_rpn,
                            random_rpn=args.random_rpn)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb.classes, 152, pretrained=True,
                            use_caffe=False,
                            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    # show model structure
    print(fasterRCNN)

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    # tr_momentum = cfg.TRAIN.MOMENTUM
    # tr_momentum = args.momentum

    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

    if args.cuda:
        fasterRCNN.cuda()

    # optimizer
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    elif args.optimizer == "sgd_decay":
        optimizer = optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM,
                              weight_decay=0.0001)

    else:
        raise NotImplementedError

    # scheduler
    if args.scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=iters_per_epoch,
                                                         eta_min=0,
                                                         last_epoch=-1)

    if args.resume:
        load_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        fasterRCNN.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)

    if args.use_tfboard:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join('logs', args.pro_name, 'pretrain')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = SummaryWriter(log_dir=log_dir)

    global_step = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        match_box_hist = None
        start = time.time()

        if not args.grad_stop and epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            with torch.no_grad():
                im_data_aug_1.resize_(data[0].size()).copy_(data[0])
                im_data_aug_2.resize_(data[1].size()).copy_(data[1])
                im_info.resize_(data[2].size()).copy_(data[2])
                gt_boxes.resize_(data[3].size()).copy_(data[3])
                num_boxes.resize_(data[4].size()).copy_(data[4])

            fasterRCNN.zero_grad()
            output = fasterRCNN(im_data_aug_1, im_data_aug_2,
                                im_info, gt_boxes, num_boxes)

            if args.random_rpn:
                contrastive_loss, rpn_loss_cls, rpn_loss_bbox = output
                losses = contrastive_loss.mean() + rpn_loss_bbox.mean() + rpn_loss_cls.mean()
            elif args.share_rpn:
                losses = output
            else:
                losses = output[0]
                num_of_matched_boxes = output[1]

            loss = losses.mean()
            loss_temp += loss.item()
            if args.use_tfboard:
                # add loss to tensorboard
                loss_item = loss.item()
                logger.add_scalar('loss_per_step', loss_item,
                                  global_step=global_step)
                # add the mean of matched box num
                if args.random_rpn:
                    contrastive_loss = contrastive_loss.mean().item()
                    rpn_loss_cls = rpn_loss_cls.mean().item()
                    rpn_loss_bbox = rpn_loss_bbox.mean().item()
                    logger.add_scalar('contrastive_loss', contrastive_loss,
                                      global_step=global_step)
                    logger.add_scalar('rpn_loss_cls', rpn_loss_cls,
                                      global_step=global_step)
                    logger.add_scalar('rpn_loss_bbox', rpn_loss_bbox,
                                      global_step=global_step)
                if not args.share_rpn:
                    mean_of_matched_boxes = num_of_matched_boxes.float().mean().item()
                    logger.add_scalar('match_box_num_per_step', mean_of_matched_boxes,
                                      global_step=global_step)

            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()
            global_step += 1

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                print("[session %d][epoch %2d][iter %4d/%4d]"
                      "loss: %.4f, lr: %.2e, time cost: %f"
                      % (args.session, epoch, step, iters_per_epoch,
                         loss_temp, lr, end - start))

                loss_temp = 0
                start = time.time()

        if args.scheduler:
            scheduler.step()

        save_name = os.path.join(output_dir,
                                 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
        save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))

    if args.use_tfboard:
        logger.close()
