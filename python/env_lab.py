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

from enum import IntEnum
import pprint
import cv2
import numpy as np

import deepmind_lab

class Action(IntEnum):
    """Action"""

    LOOK_LEFT = 0
    LOOK_RIGHT = 1
    MOVE_FORWARD = 2
    # LOOK_MOVE_LEFT =3
    # LOOK_MOVE_RIGHT =4
    # MOVE_BACK = 5

class Buffer(object):
    """ A buffer to collect observations until they form a state. """
    def __init__(self, sequence_length, width, height):
        # _logger.info("Initializing new object of type " + str(type(self).__name__))
        self.buffer = np.zeros((sequence_length,
                                width,
                                height), dtype=np.uint8)
        self.buffer_size = np.shape(self.buffer)
        
    def add(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        
    def get_state(self):
        return self.buffer
    
    def reset(self):
        self.buffer *= 0

class EnvLab(object):
    def __init__(self, level_script, height, width, channel, config):
        """Construct and start the environment."""
        self.env = deepmind_lab.Lab(level_script, ['BGR_INTERLEAVED'], config = config)
        # {
        #         "fps": str(fps),
        #         "width": str(width),
        #         "height": str(height)
        #     }
        self.env.reset()
        self.channel = channel
        self.resolution = (self.channel, height, width)
        self.sequence_length = 4

        observation_spec = self.env.observation_spec()
        print('Observation spec:')
        pprint.pprint(observation_spec)

        self.action_spec = self.env.action_spec()
        print('Action spec:')
        pprint.pprint(self.action_spec)

        self.indices = {a["name"]: i for i, a in enumerate(self.action_spec)}
        self.mins = np.array([a["min"] for a in self.action_spec])
        self.maxs = np.array([a["max"] for a in self.action_spec])
        self.buffer = Buffer(self.sequence_length,
                             width,
                             height)
        self.num_actions = len(Action)

        # obs = self.env.observations()  # dict of Numpy arrays
        # rgb_i = obs['RGB_INTERLEAVED']
        # print('Observation shape:', rgb_i.shape)
        # sys.stdout.flush()

    def reset(self):
        self.env.reset()
        # print('reset')
        # self.buffer = Buffer(self.sequence_length,
        #                      self.resolution[2],
        #                      self.resolution[1])

    def observations(self):
        obs = self.env.observations()
        img = obs['BGR_INTERLEAVED']
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    
    def print_step(self, obs):
        obs = self.env.observations()
        img = obs['DEBUG.CAMERA_INTERLEAVED.TOP_DOWN']
        # cv2.circle(img, (120, 50), 3, (0,255,0), -1)
        cv2.imshow('map', img)
        cv2.waitKey(2)

    def Preprocess(self, img):
        #cv2.imshow("frame-train", img)
        #cv2.waitKey(20)
        # resolution = (self.height, self.width, self.channel)
        img = cv2.resize(img, (self.resolution[2], self.resolution[1]))
        if (self.channel == 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        # cv2.imshow("frame-test", img)
        # cv2.waitKey(20)
        # print("img:", img.shape)
        return img.transpose([2, 0, 1])
    
    def Preprocess_gray(self, img):
        #cv2.imshow("frame-train", img)
        #cv2.waitKey(20)
        # resolution = (self.height, self.width, self.channel)
        img = cv2.resize(img, (self.resolution[2], self.resolution[1]))
        if (self.channel == 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("frame-test", img)
        # cv2.waitKey(20)
        # print("img:", img.shape)
        return img
    
    def Preprocess_stack(self, img):
        img = cv2.resize(img, (self.resolution[2], self.resolution[1]))
        if (self.channel == 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)
        
        img = img.transpose([2, 0, 1])
        self.buffer.add(img)
        img = self.buffer.get_state()
        # print('0:', img[0])
        # print('3:', img[3])
        # cv2.waitKey(1000)
        return img

    def numActions(self):
        return self.num_actions #self.num_actions*2
    
    # def getAction(self):
    #     return np.random.choice(range(self.num_actions))

    def is_running(self):
        return self.env.is_running()
    
    def mapping(self, action_raw):
        self.action = np.zeros([len(self.action_spec)])

        if (action_raw == Action.LOOK_LEFT):
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = -30
        elif (action_raw == Action.LOOK_RIGHT):
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = 30
        elif (action_raw == Action.MOVE_FORWARD):
            self.action[self.indices["MOVE_BACK_FORWARD"]] = 1
        elif (action_raw == Action.LOOK_MOVE_LEFT):
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = -25
            self.action[self.indices["MOVE_BACK_FORWARD"]] = 1
        elif (action_raw == Action.LOOK_MOVE_RIGHT):
            self.action[self.indices["LOOK_LEFT_RIGHT_PIXELS_PER_FRAME"]] = 25
            self.action[self.indices["MOVE_BACK_FORWARD"]] = 1

        return np.clip(self.action, self.mins, self.maxs).astype(np.intc)
    
    def step(self, action, num_steps = 4):
        # frame_repeat = 4
        action = self.mapping(action)
        pick_reward = self.env.step(action, num_steps = num_steps)
        
        is_terminal = not self.is_running() #or pick_reward == 10
        # if pick_reward == 0:
        #     penalty = 0 # -1
        # else: penalty = 0
        
        # for _ in range(num_steps):
        #     reward = self.env.step(action, 1)
        #     pick_reward += reward
        #     is_terminal = not self.is_running() or pick_reward == 10

        #     if (is_terminal):
        #         break

        reward = pick_reward # + penalty
        return reward, is_terminal

