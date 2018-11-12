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

import numpy as np
import pandas as pd


class QLearning(object):
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # action is a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, state):
        self.check_state(state)
        # select an action
        if np.random.uniform() < self.epsilon:
            # get all possible good actions with in current state
            state_action = self.q_table.loc[state, :]
            # randomly choose one in the best actions
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index
            )
        else:
            # choose a random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != "done":  # next state is not final state
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # next state is final state

        self.q_table.loc[s, a] += self.lr * \
            (q_target - q_predict)  # update q_table

    def check_state(self, state):
        if state not in self.q_table.index:
            # add new state in to q_table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions), index=self.q_table.columns, name=state
                )
            )

    def save_q_table(self):
        self.q_table.to_csv('learned_q_table.csv')
