import numpy as np
from skimage import io as skio
from feature import img_feature_analysis as ifa
import matplotlib.pyplot as plt

FOCAL_UP = 199
FOCAL_DOWN = 0


class CameraEnv(object):
    viewer = None
    action_bound = [-20, 20]
    state_dim = 2
    action_dim = 1

    def __init__(self):
        self.action_space = ["lf", "sf", "hp", "sb", "lb"]
        self.n_actions = len(self.action_space)
        self.focal_old = 0
        self.focal_new = 0
        self.focused = 0
        self.img_stack = skio.imread("data/Res256/train.tif")
        self.max_grey = 0

    def step(self, action):
        done = False
        r = 0
        exceed_bound = False
        area_sel_old = 0
        grey_sum_old = 0
        grey_avg_old = 0
        threshold_old = 0
        grey_std_old = 0

        area_sel_new = 0
        grey_sum_new = 0
        grey_avg_new = 0
        threshold_new = 0
        grey_std_new = 0

        if action == 0:   # large forward step
            if self.focal_old < (FOCAL_UP-10):
                self.focal_new = int(self.focal_old + 10)
            else:
                exceed_bound = True
        elif action == 1:   # small forward step
            if self.focal_old < (FOCAL_UP-1):
                self.focal_new = int(self.focal_old + 1)
            else:
                exceed_bound = True
        elif action == 2:   # hold focal position
            self.focal_new = self.focal_old
        elif action == 3:   # small backward step
            if self.focal_old > (FOCAL_DOWN+1):
                self.focal_new = int(self.focal_old - 1)
            else:
                exceed_bound = True
        elif action == 4:   # large backward step
            if self.focal_old > (FOCAL_DOWN+10):
                self.focal_new = int(self.focal_old - 10)
            else:
                exceed_bound = True

        if (exceed_bound):
            done = False
            r -= 5
            # state
            # normalize features
            area_sel_old, grey_sum_old, grey_avg_old, threshold_old, grey_std_old = ifa(
                self.img_stack[self.focal_old]
            )
            s = self.focal_new
        else:
            # normalize features
            area_sel_old, grey_sum_old, grey_avg_old, threshold_old, grey_std_old = ifa(
                self.img_stack[self.focal_old]
            )
            area_sel_new, grey_sum_new, grey_avg_new, threshold_new, grey_std_new = ifa(
                self.img_stack[self.focal_new]
            )

            r += (grey_avg_new - grey_avg_old) * grey_avg_new / 50

            # done and reward
            if (self.focal_new == self.focal_old) and (grey_avg_new == self.max_grey):
                r += 0.001
                self.focused += 1
                if self.focused > 20:
                    done = True
                    print('hold position at focal plane: ', self.focal_new)
            else:
                self.focused = 0

            # state
            s = self.focal_new

        if grey_avg_new > grey_avg_old:
            self.max_grey = grey_avg_new

        self.focal_old = self.focal_new

        return s, r, done

    def reset(self):
        self.focal_new = int(np.random.rand() * 200)
        self.focal_old = int(np.random.rand() * 200)
        self.focused = 0

        area_sel_new, grey_sum_new, grey_avg_new, threshold_new, grey_std_new = ifa(
            self.img_stack[self.focal_new]
        )
        # state
        s = self.focal_new
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.img_stack, self.focal_new)
        self.viewer.render(self.focal_new)

    def sample_action(self):
        return int(np.random.rand() * 4)


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
