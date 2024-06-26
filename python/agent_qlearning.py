## Copyright (C) 2016-17 Google Inc.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
################################################################################
"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
import sys
import numpy as np
import six
import cv2
import random
import time
import pickle
from collections import defaultdict
from tqdm import trange

import deepmind_lab

from env_lab import EnvLab

class ExponentialSchedule:
    def __init__(self, value_from, value_to, num_steps):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: initial value
        :param value_to: final value
        :param num_steps: number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        # determine the `a` and `b` parameters such that the schedule is correct
        self.a = self.value_from
        self.b = np.log(self.value_to / self.a) / (self.num_steps - 1)

    def value(self, step) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step:  The step at which to compute the interpolation.
        :rtype: float.  The interpolated value.
        """

        # implement the schedule rule as described in the docstring,
        # using attributes `self.a` and `self.b`.
        if step <=0:
            value = self.value_from
        
        elif step >= (self.num_steps - 1):
            value = self.value_to
            
        else:
            value = self.a * np.exp(self.b * step)
        
        return value
    
class QLearningAgent:
    def __init__(self, state_size, env):
        self.Q = defaultdict(lambda: np.zeros(env.NumActions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-l', '--level_script', type=str,
                        default='seekavoid_arena_01',
                        help='The level that is to be played. Levels'
                        'are Lua scripts, and a script called \"name\" means that'
                        'a file \"assets/game_scripts/name.lua is loaded.')
    parser.add_argument('-s', '--level_settings', type=str, default=[],
                        action='append',
                        help='Applies an opaque key-value setting. The setting is'
                        'available to the level script. This flag may be provided'
                        'multiple times. Universal settings are `width` and '
                        '`height` which give the screen size in pixels, '
                        '`fps` which gives the frames per second, and '
                        '`random_seed` which can be specified to ensure the '
                        'same content is generated on every run.')
    # parser.add_argument('--runfiles_path', type=str, default=None,
    #                     help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--num_steps', type=int, default=1,
                        help='The number of episodes to play.')
    args = parser.parse_args()

  # Convert list of level setting strings (of the form "key=value") into a
  # `config` key/value dictionary.
    config = {
        k: v
        for k, v in [six.ensure_str(s).split('=') for s in args.level_settings]
    }

    # if args.runfiles_path:
    #     deepmind_lab.set_runfiles_path(args.runfiles_path)

    observation_input = (240, 320, 3)
    channel = 1
    gamma = 0.9
    step_size = 0.5
    env = EnvLab(args.level_script, 84, 84, channel, config)
    eploration = ExponentialSchedule(1.0, 0.01, 1_000_000)

    Q = defaultdict(lambda: np.zeros(env.numActions()))

    # policy = create_epsilon_policy(Q,epsilon)

    # returns = np.zeros(num_episodes)
    num_step = np.zeros(args.num_steps)
    state = env.Preprocess_gray(env.observations())
    rewards = 0

    for i in trange(args.num_steps, desc="Num_steps", leave=False):
        # TODO Q4
        # For each episode calculate the return
        # Update Q
        prob = np.random.random()
        if prob > 0.1:
            print("exploitation")
            # state_key = tuple(state)
            state_key = tuple(map(tuple, state))
            action = np.random.choice(np.where(Q[state_key] == Q[state_key].max()))
        # Else doing a random choice --> exploration
        else:
            print("exploration")
            action = np.array(range(env.numActions()))

        print(action)
        reward, done = env.step(action)
        next_pos = env.Preprocess_gray(env.observations())
        # next_action = policy(next_pos)
        Q[state][action] += step_size * (reward + gamma * Q[next_pos].max() - Q[state][action])
        rewards += reward
        
        state = next_pos

        if done:
            print(rewards)
            rewards = 0
            state = env.reset()
        #     num_episode[i] = num_episode[i - 1] + 1
        # else: num_episode[i] = num_episode[i - 1]

    # agent = QLearningAgent(observation_input, env)
    # # Create an action to move forwards.
    # step_size = 0.5
    # gamma = 0.99
    # for _ in six.moves.range(args.num_episodes):
    #     state_size = env.Preprocess(env.observations()).shape
    #     state = env.Preprocess(state_raw)
    #     print(state)

    #     step = 0
    #     done = False
    #     total_rewards = 0
    #     while env.is_running():
    #         # Advance the environment 4 frames while executing the action.
    #         # reward = env.step(action, 4)

    #         # if reward != 0:
    #         #     score += reward
    #         #     obs = env.observations()  # dict of Numpy arrays
    #         #     # rgb_i = obs['RGB_INTERLEAVED']
    #         #     print('Observation shape:', obs.shape)
    #         prob = np.random.random()
    #         if prob > 0.1:
    #             print("exploitation")
    #             action = np.random.choice(np.where(agent.Q[state] == agent.Q[state].max())[0])
    #         # Else doing a random choice --> exploration
    #         else:
    #             print("exploration")
    #             action = np.array(range(env.NumActions()))
            
    #         reward = env.step(action, num_steps=4)
    #         next_state = env.observations()

    #         agent.Q[state][action] += step_size * (reward + gamma * agent.Q[next_state].max() - agent.Q[state][action])

    #         state = next_state
