# -*- coding: utf-8 -*-

# by Dr. Ming Yan (11/2018)
# yan.meen@gmail.com
# https://github.com/yanmeen/rlaf
#
# =============================================================================
from env import CameraEnv
from RL_sarsa import SarsaLambdaTable

MAX_EPISODES = 3000
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = CameraEnv()

# set RL method
RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

steps = []


def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()

        # RL choose action based on observation
        a = RL.choose_action(str(s))

        # initial all zero eligibility trace
        RL.eligibility_trace *= 0

        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            # RL take action and get next observation and reward
            s_, reward, done = env.step(a)

            # RL choose action based on next observation
            a_ = RL.choose_action(str(s_))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(s), a, reward, str(s_), a_)

            # swap observation and action
            s = s_
            a = a_

            ep_r += reward

            if done or j == MAX_EP_STEPS - 1:
                print(
                    "Ep: %i | %s | ep_r: %.1f | step: %i"
                    % (i, "---" if not done else "done", ep_r, j)
                )
                break


def eval():
    # env.render()
    while True:
        s = env.reset()
        done = 0
        print("start from focal position: ", env.focal_new)
        while not done:
            # env.render()
            a = RL.choose_action(s)
            s, r, done = env.step(a)
            print("goes to new focal position: ", env.focal_new)
            if done:
                print("========>>>>>>>>>>>>> Find focus point at: ", env.focal_new)


if ON_TRAIN:
    train()
else:
    eval()
