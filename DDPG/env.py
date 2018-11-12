# -*- coding: utf-8 -*-

# by Dr. Ming Yan (11/2018)
# yan.meen@gmail.com
# https://github.com/yanmeen/rlaf
#
# =============================================================================
import numpy as np
from skimage import io as skio
from feature import img_feature_analysis as ifa
import matplotlib.pyplot as plt

FOCAL_UP = 199
FOCAL_DOWN = 0


class CameraEnv(object):
    viewer = None
    action_bound = [-10, 10]
    state_dim = 2
    action_dim = 1

    def __init__(self):
        self.focal_old = 0
        self.focal_new = 0
        self.focused = 0
        self.img_stack = skio.imread("data/Res256/train1.tif")
        self.max_grey = 0
        self.hist_focal = np.zeros(self.state_dim, dtype=np.int32)

    def step(self, action):
        done = False
        #action = np.clip(action-self.action_bound[1], *self.action_bound)
        r = 0

        self.focal_new = int(
            np.clip((self.focal_old + action), FOCAL_DOWN, FOCAL_UP))

        for i in range(0, self.state_dim - 1):
            if i < self.state_dim - 2:
                self.hist_focal[i] = self.hist_focal[i + 1]
            if i == self.state_dim - 2:
                self.hist_focal[i] = self.focal_new

        if (self.focal_new == FOCAL_UP) or (self.focal_new == FOCAL_DOWN):
            r -= 5
            # normalize features
            grey_avg_old = 0
            grey_avg_new = 0

        else:
            # normalize features
            grey_avg_old = ifa(self.img_stack[self.focal_old])
            grey_avg_new = ifa(self.img_stack[self.focal_new])

            # done and reward
            if (self.focal_new == self.focal_old) and (grey_avg_new * 255 == self.max_grey):
                self.focused += 1
                if self.focused > 20:
                    self.hist_focal[self.state_dim - 1] = 1
                    r += 10
                    done = True
                    self.focused = 0
                    print("hold position at focal plane for 20 times: ",
                          self.focal_new)
                else:
                    r -= 0.01
            else:
                self.focused = 0
                # r -= np.sum(np.sqrt(np.asarray(feature_new) -
                #                    np.asarray(feature_old)))
                r += (grey_avg_new - grey_avg_old) * 255

            # state

        #s = self.hist_focal
        s = np.asarray([self.focal_new, 1 if done else 0])

        self.focal_old = self.focal_new
        if grey_avg_new > grey_avg_old:
            self.max_grey = grey_avg_new * 255

        return s, r, done

    def reset(self):
        self.focal_new = int(np.random.rand() * 200)
        self.focal_old = self.focal_new
        self.focused = 0
        self.hist_focal = np.zeros(self.state_dim, dtype=np.int32)

        # self.hist_focal = np.random.randint(0, high=199, size=self.state_dim)
        self.hist_focal[self.state_dim - 1] = 0

        # state
        # s = self.hist_focal
        s = np.asarray([self.focal_new, 0])
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.img_stack, self.focal_new)
        self.viewer.render(self.focal_new)

    def sample_action(self):
        return int(np.random.rand() * 20)


class Viewer(object):
    def __init__(self, img_stack, focal):
        self.img_stack = img_stack
        self.focal = focal

        plt.imshow(self.img_stack[self.focal])
        plt.show()

    def render(self, focal):
        self.focal = focal
        plt.imshow(self.img_stack[self.focal])
        plt.show()


if __name__ == "__main__":
    env = CameraEnv()
    while True:
        env.render()
        s, r, done = env.step(env.sample_action())
