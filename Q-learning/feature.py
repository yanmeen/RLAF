# -*- coding: utf-8 -*-

# by Dr. Ming Yan (10/2018)
# yan.meen@gmail.com
# https://github.com/rlaf
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


def img_feature_analysis(img, bkgrd=8):

    img = np.array(img, dtype="float32") / 255
    bkgrd = np.median(img)

    img[img <= bkgrd] = 0
    img_grey = img

    max_grey = np.max(img_grey[img_grey != 0])
    threshold = max_grey * 0.618
    # threshold = np.mean(img_grey[img_grey != 0])
    # threshold = 2 * bkgrd
    # print('Threshold is: ', threshold)

    # img_bin = np.where(img_grey > threshold, upper, lower)
    area_sel = np.sum(img_grey > threshold)

    img_sel = img_grey[img_grey > threshold]

    grey_sum = np.sum(img_sel)
    grey_avg = grey_sum / area_sel
    grey_std = np.std(img_grey)
    return grey_avg  # (area_sel, grey_sum, grey_avg, threshold, grey_std)


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
                    x_feature = np.array(img_feature_analysis(x, bkgrd=10))
                    y_feature = np.array(img_feature_analysis(y, bkgrd=10))

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
