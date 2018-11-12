# -*- coding: utf-8 -*-

# by Dr. Ming Yan (11/2018)
# yan.meen@gmail.com
# https://github.com/yanmeen/rlaf
#
# =============================================================================
# Q-learning code for reinforcement autofocus
# action space: [ff, sf, hp, sb, fb]
#   fast foward, slow forward, hold position, slow backward, fast backword
# focal +5,     +1,     0,      -1,     -5
# state space: integer, [focal_down, focal_up]
# learn and search state space with Q-table
# select an action from current state to best focal point
# to maximize the total reward

from camera import CameraEnv
from rl_Q import QLearning

MAX_EPISODE = 500
MAX_EP_STEP = 50

env = CameraEnv()
RL = QLearning(actions=list(range(env.n_actions)))

steps = []


def train():
    for ep in range(MAX_EPISODE):
        # initial state
        state = env.reset()
        ep_reward = 0.

        for i in range(MAX_EP_STEP):

            # choose action based on current state
            action = RL.choose_action(str(state))

            # take action and get next state and reward
            state_, reward, done = env.step(action)
            ep_reward += reward

            # learn from the state transition
            RL.learn(str(state), action, reward, str(state_))

            # update state
            state = state_

            # break the loop if done
            if done or i == MAX_EP_STEP - 1:
                print(
                    "Ep: %i | %s | ep_reward: %.1f | step: %i | stop at: %i | max_grey: %i"
                    % (
                        ep,
                        "---" if not done else "done",
                        ep_reward,
                        i,
                        state,
                        env.max_grey,
                    )
                )
                break

    print("max episode reached")
    print(len(RL.q_table))
    RL.save_q_table()


train()
