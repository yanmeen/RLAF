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
    action_dim = 5

    def __init__(self):
        self.action_space = ["ff", "sf", "hp", "sb", "fb"]
        self.n_actions = len(self.action_space)
        self.focal_old = 0
        self.focal_new = 0
        self.focused = 0
        self.img_stack = skio.imread("data/Res256/train1.tif")
        self.max_grey = 0
        #self.hist_focal = np.zeros(self.state_dim, dtype=np.int32)

    def step(self, a):
        done = False
        reward = 0
        exceed_bound = False
        action = np.argmax(a)

        if action == 0:  # large forward step
            if self.focal_old < (FOCAL_UP - 10):
                self.focal_new = int(self.focal_old + 10)
            else:
                exceed_bound = True
        elif action == 1:  # small forward step
            if self.focal_old < (FOCAL_UP - 1):
                self.focal_new = int(self.focal_old + 1)
            else:
                exceed_bound = True
        elif action == 2:  # hold focal position
            self.focal_new = self.focal_old
        elif action == 3:  # small backward step
            if self.focal_old > (FOCAL_DOWN + 1):
                self.focal_new = int(self.focal_old - 1)
            else:
                exceed_bound = True
        elif action == 4:  # large backward step
            if self.focal_old > (FOCAL_DOWN + 10):
                self.focal_new = int(self.focal_old - 10)
            else:
                exceed_bound = True

        if exceed_bound:
            reward -= 5
            grey_avg_new = 0
            grey_avg_old = 0
        else:
            grey_avg_new = ifa(self.img_stack[self.focal_new])
            grey_avg_old = ifa(self.img_stack[self.focal_old])

            if (self.focal_new == self.focal_old) and (grey_avg_new*255 == self.max_grey):
                # reward += 1
                self.focused += 1
                if self.focused > 20:
                    done = True
                    self.focused = 0
                    reward += 10
                    print("hold the position at focal plane 20 times:",
                          self.focal_new)
                else:
                    reward -= 0.01
            else:
                self.focused = 0
                # np.sign(grey_avg_new - grey_avg_old)
                reward += (grey_avg_new - grey_avg_old) * 255

        #next_state = self.focal_new
        s = np.asarray([self.focal_new, 1 if done else 0])

        if grey_avg_new > grey_avg_old:
            self.max_grey = grey_avg_new*255

        self.focal_old = self.focal_new

        return s, reward, done

    def reset(self):
        self.focal_new = int(np.random.rand() * 200)
        self.focal_old = self.focal_new
        self.focused = 0
        # self.hist_focal = np.zeros(self.state_dim, dtype=np.int32)

        # self.hist_focal = np.random.randint(0, high=199, size=self.state_dim)
        #self.hist_focal[self.state_dim - 1] = 0

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
