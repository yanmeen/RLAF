"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""
from env_Camera import CameraEnv
from rl import DDPG

MAX_EPISODES = 10000
MAX_EP_STEPS = 100
ON_TRAIN = True

# set env
env = CameraEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []


def train():
    # start training
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for i in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            # break the loop if done
            if done or i == MAX_EP_STEPS - 1:
                print(
                    "Ep: %i | %s | ep_reward: %.1f | step: %i | stop at: %i | max_grey: %i"
                    % (ep, "---" if not done else "done", ep_r, i, s[0], env.max_grey)
                )
                break
    rl.save()


def eval():
    rl.restore()
    # env.render()
    while True:
        s = env.reset()
        done = 0
        print("start from focal position: ", env.focal_new)
        while not done:
            # env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            print("goes to new focal position: ", env.focal_new)
            if done:
                print("========>>>>>>>>>>>>> Find focus point at: ", env.focal_new)


if ON_TRAIN:
    train()
else:
    eval()
