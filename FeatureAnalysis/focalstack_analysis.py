# -*- coding: utf-8 -*-

# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
# https://github.com/rlaf
# modified on the code from https://github.com/cszn
# =============================================================================

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0, 0.9] rather than the default
# =============================================================================

import argparse
import time
import os
import glob
import math
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from keras.utils import np_utils

# import keras.backend as K


def parse_args():
    # Params
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set_dir", default="data/Res256", type=str, help="path of train data"
    )
    parser.add_argument(
        "--set_names",
        default=["NEW"],  # "SM", "MM", "LM"],  # , "DM"],
        type=list,
        help="name of test dataset",
    )
    parser.add_argument(
        "--result_dir",
        default="ana_results",
        type=str,
        help="directory of analysis results",
    )
    return parser.parse_args()


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def img_analysis(img, bkgrd=8):

    img = np.array(img, dtype="float32") / 255
    bkgrd = np.median(img)

    img[img <= bkgrd] = 0
    img_grey = img

    max_grey = np.max(img_grey[img_grey != 0])
    min_grey = np.min(img_grey[img_grey != 0])
    threshold = max_grey * 0.618
    # threshold = np.mean(img_grey[img_grey != 0])
    # threshold = 2 * bkgrd
    # print('Threshold is: ', threshold)

    # img_bin = np.where(img_grey > threshold, upper, lower)
    area_sel = np.sum(img_grey > threshold)

    img_sel = img_grey[img_grey > threshold]
    # sq_area = area ** 2
    # sqrt_area = math.sqrt(area)

    grey_sum = np.sum(img_sel)
    grey_total = np.sum(img_grey)
    # sq_sum_grey = sum_grey ** 2
    # sqrt_sum_grey = math.sqrt(sum_grey)
    grey_avg = grey_sum / area_sel

    grey_std = np.std(img_sel)
    # bright_pixel = np.sum(img_grey[img_grey > 0.9*max_grey])
    # dark_pixel = np.sum(img_grey[img_grey < 1.1*min_grey])
    bright_pixel = max_grey
    dark_pixel = min_grey

    b2d_ratio = bright_pixel / dark_pixel

    return (
        area_sel,
        grey_sum,
        grey_avg,
        threshold,
        grey_std,
        bright_pixel,
        dark_pixel,
        b2d_ratio,
        grey_total,
    )


if __name__ == "__main__":

    args = parse_args()

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    for set_cur in args.set_names:
        if not os.path.exists(os.path.join(args.result_dir, set_cur)):
            os.mkdir(os.path.join(args.result_dir, set_cur))

        save_dir = os.path.join(args.result_dir, set_cur)

        for im in os.listdir(os.path.join(args.set_dir, set_cur)):
            if im.endswith(".tif"):
                imgs = skio.imread(
                    os.path.join(args.set_dir, set_cur, im)
                )  # gray scale
                focal_number = imgs.shape[0]

                ana_result = []
                ana_diff = []
                start_time = time.time()
                for id in range(0, focal_number - 1):
                    x = np.array(imgs[id], dtype=np.float32)
                    y = np.array(imgs[id + 1], dtype=np.float32)
                    x_feature = np.array(img_analysis(x, bkgrd=10))
                    y_feature = np.array(img_analysis(y, bkgrd=10))

                    ana_result.append(x_feature)
                    if id == (focal_number - 2):
                        ana_result.append(y_feature)

                elapsed_time = time.time() - start_time
                print("%10s : %10s : %2.4f second" %
                      (set_cur, im, elapsed_time))

            ana_result = np.array(ana_result, dtype="float32")
            np.savetxt(
                os.path.join(save_dir, im + "_ana.csv"), ana_result, delimiter=","
            )
