#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains, evaluates and saves the model network using a queue."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
from PIL import Image
import commentjson

from stwn.modules.evaluation.kitti_devkit import seg_utils as seg

import time

import stwn.modules.tensorvision.utils as utils


def readdataset(hypes):
    data_dir = hypes['dirs']['data_dir']
    data_file = hypes['data']['val_file']
    data_file = os.path.join(data_dir, data_file)
    files = [line.rstrip() for line in open(data_file)]
    return files


def Vocevalseg(hypes):

    num = hypes['arch']['num_classes'] + 1
    confcounts = np.zeros(num)
    count = 0
    data_dir = hypes['dirs']['data_dir']
    dataset = readdataset(hypes)

    for i in dataset:
        image_file = 'JPEGImages/' + i + '.jpg'
        image_file = os.path.join(data_dir, image_file)
        assert os.path.exists(image_file), \
            "File does not exist: %s" % image_file
        gt_file = 'SegmentationClass/' + i + '.png'
        gt_file = os.path.join(data_dir, gt_file)
        assert os.path.exists(gt_file), \
                "File does not exist: %s" % gt_file
        image = np.array(Image.open(image_file))
        gt_image = np.array(Image.open(gt_file))[..., np.newaxis]





def eval_image(hypes, gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0

    road_color = np.array(hypes['data']['road_color'])
    background_color = np.array(hypes['data']['background_color'])
    gt_road = np.all(gt_image == road_color, axis=2)
    gt_bg = np.all(gt_image == background_color, axis=2)
    valid_gt = gt_road + gt_bg

    FN, FP, posNum, negNum = seg.evalExp(gt_road, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def evaluate(hypes, sess, image_pl, inf_out):

    softmax = inf_out['softmax']
    data_dir = hypes['dirs']['data_dir']

    eval_dict = {}
    for phase in ['val']:
        data_dir = hypes['dirs']['data_dir']
        dataset = readdataset(hypes)

        thresh = np.array(range(0, 256))/255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        total_posnum = 0
        total_negnum = 0

        image_list = []

        for i in dataset:
            image_file = 'JPEGImages/' + i + '.jpg'
            image_file = os.path.join(data_dir, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file
            gt_file = 'SegmentationClass/' + i + '.png'
            gt_file = os.path.join(data_dir, gt_file)
            assert os.path.exists(gt_file), \
                "File does not exist: %s" % gt_file
            image = np.array(Image.open(image_file))
            gt_image = np.array(Image.open(gt_file))[..., np.newaxis]

            input_image = image

            shape = input_image.shape

            feed_dict = {image_pl: input_image}

            output = sess.run([softmax], feed_dict=feed_dict)
            output_im = output[0][:, 1].reshape(shape[0], shape[1])


            if phase == 'val':
                # Saving RB Plot
                ov_image = seg.make_overlay(image, output_im)
                name = os.path.basename(image_file)
                image_list.append((name, ov_image))

                name2 = name.split('.')[0] + '_green.png'

                hard = output_im > 0.5
                green_image = utils.fast_overlay(image, hard)
                image_list.append((name2, green_image))

            FN, FP, posNum, negNum = eval_image(hypes,
                                                gt_image, output_im)

            total_fp += FP
            total_fn += FN
            total_posnum += posNum
            total_negnum += negNum

        eval_dict[phase] = seg.pxEval_maximizeFMeasure(
            total_posnum, total_negnum, total_fn, total_fp, thresh=thresh)

        if phase == 'val':
            start_time = time.time()
            for i in range(10):
                sess.run([softmax], feed_dict=feed_dict)
            dt = (time.time() - start_time)/10

    eval_list = []

    for phase in ['val']:
        eval_list.append(('[{}] MaxF1'.format(phase),
                          100*eval_dict[phase]['MaxF']))
        eval_list.append(('[{}] BestThresh'.format(phase),
                          100*eval_dict[phase]['BestThresh']))
        eval_list.append(('[{}] Average Precision'.format(phase),
                          100*eval_dict[phase]['AvgPrec']))
    eval_list.append(('Speed (msec)', 1000*dt))
    eval_list.append(('Speed (fps)', 1/dt))

    return eval_list, image_list


def main():
    with open('/home/lanhai/Projects/FCN/stwn/hypes/Pascal.json') as f:
        hypes = commentjson.load(f)

    Vocevalseg(hypes)

if __name__ == '__main__':
    main()