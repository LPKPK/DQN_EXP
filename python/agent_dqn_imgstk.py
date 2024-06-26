#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import random
import time
import sys
import os
import matplotlib.pyplot as plt
from collections import namedtuple
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

def MakeDir(path):
    try:
        os.makedirs(path)
    except:
        pass

def rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

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

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)

class ReplayMemory:
    def __init__(self, max_size, resolution):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer.
            - state_size: Size of the state-space features for the environment.
        """
        self.max_size = max_size
        self.state_size = resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")


        # preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size,) + resolution).to(self.device)
        self.actions = torch.empty((max_size, 1), dtype=torch.long).to(self.device)
        self.rewards = torch.empty((max_size, 1)).to(self.device)
        self.next_states = torch.empty((max_size,) + resolution).to(self.device)
        self.dones = torch.empty((max_size, 1), dtype=torch.bool).to(self.device)

        # pointer to the current location in the circular buffer
        self.idx = 0
        # indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        :param state:  1-D np.ndarray of state-features.
        :param action:  integer action.
        :param reward:  float reward.
        :param next_state:  1-D np.ndarray of state-features.
        :param done:  boolean value indicating the end of an episode.
        """

        # store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`

        self.states[self.idx] = torch.as_tensor(state).to(self.device)
        self.actions[self.idx] = torch.as_tensor(action).to(self.device)
        self.rewards[self.idx] = torch.as_tensor(reward).to(self.device)
        self.next_states[self.idx] = torch.as_tensor(next_state).to(self.device)
        self.dones[self.idx] = torch.as_tensor(done).to(self.device)
        
        # DO NOT EDIT
        # circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :rtype: Batch
        """

        # randomly sample an appropriate number of
        # transitions *without replacement*.  If the buffer contains less than
        # `batch_size` transitions, return all of them.  The return type must
        # be a `Batch`.

        if self.size < batch_size:
            batch = Batch(self.states, self.actions, self.rewards, self.next_states, self.dones)
        else:
            sample_indices = np.random.choice(self.size, batch_size, replace = False)
            batch = Batch(self.states[sample_indices], self.actions[sample_indices], self.rewards[sample_indices], 
                          self.next_states[sample_indices], self.dones[sample_indices])

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env:  Openai Gym environment
        :param num_steps:  Number of steps to populate the
        """

        # run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        # Hint:  don't repeat code!  Use the self.add() method!

        env.reset()
        state = env.Preprocess_stack(env.observations())
    
        for i in range(num_steps):
            action = np.random.choice(env.numActions())
            reward, done = env.step(action)
            # done = not env.is_running()
            
            if done:
                env.reset()
                next_state = np.zeros(self.state_size)
                self.add(state, action, reward, next_state, done)
                state = env.Preprocess_stack(env.observations())
            # env.print_step(env.observations())
            else:
                next_state = env.Preprocess_stack(env.observations())
                self.add(state, action, reward, next_state, done)
                state = next_state
            # print(i, state.shape)
                

class DQN(nn.Module):
    def __init__(self, resolution, action_dim, *, num_layers=4, hidden_dim=256):
        """Deep Q-Network PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.state_dim = resolution
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # define the layers of your model such that
        # * there are `num_layers` nn.Linear modules / layers
        # * all activations except the last should be ReLU activations
        #   (this can be achieved either using a nn.ReLU() object or the nn.functional.relu() method)
        # * the last activation can either be missing, or you can use nn.Identity()
        
        # layer5
        # self.conv1 = nn.Conv2d(in_channels=resolution[0], out_channels=16, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        # self.fc1 = nn.Linear(32 * 7 * 7, self.hidden_dim)  # For not 40 * 40, Adjust the input size based on your resolution
        # self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)

        # self.conv1 = nn.Conv2d(in_channels=resolution[0], out_channels=8, kernel_size=3, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2)
        # self.fc1 = nn.Linear(16 * 9 * 9, self.hidden_dim)  # For not 40 * 40, Adjust the input size based on your resolution
        # self.fc2 = nn.Linear(self.hidden_dim, self.action_dim)

        # lenet
        self.layer1 = nn.Sequential(
            nn.Conv2d(resolution[0], 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(18 * 18 * 16, 128)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, self.action_dim)

    def forward(self, x) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space.
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space.  This represents the Q values Q(s, .).
        """
        # use the defined layers and activations to compute
        # the action-values tensor associated with the input states.

        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = x.view(x.size(0), -1)  # Flatten the output for fully connected layer
        # # print("x:", x.size(0))
        # x = F.relu(self.fc1(x))
        # q_values = self.fc2(x)

        # LeNet
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        
        return out
        # return q_values

    # utility methods for cloning and storing models.  DO NOT EDIT
    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            },
            'state_dict': self.state_dict(),
        }
    
def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    """Perform a single batch-update step on the given DQN model.

    :param optimizer: nn.optim.Optimizer instance.
    :param batch:  Batch of experiences (class defined earlier).
    :param dqn_model:  The DQN model to be trained.
    :param dqn_target:  The target DQN model, ~NOT~ to be trained.
    :param gamma:  The discount factor.
    :rtype: float  The scalar loss associated with this batch.
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
    """
    # YOUR CODE HERE:  compute the values and target_values tensors using the
    # given models and the batch of data.
    values = dqn_model(batch.states).gather(1, batch.actions)
    
    next_value = torch.max((~batch.dones) * dqn_target(batch.next_states), 1)[0].detach()
    
    next_value = torch.unsqueeze(next_value, 1)
    target_values = batch.rewards + gamma * next_value

    # DO NOT EDIT FURTHER

    assert (
        values.shape == target_values.shape
    ), 'Shapes of values tensor and target_values tensor do not match.'

    # testing that the value tensor requires a gradient,
    # and the target_values tensor does not
    assert values.requires_grad, 'values tensor should not require gradients'
    assert (
        not target_values.requires_grad
    ), 'target_values tensor should require gradients'

    # computing the scalar MSE loss between computed values and the TD-target
    loss = F.mse_loss(values, target_values)

    optimizer.zero_grad()  # reset all previous gradients
    loss.backward()  # compute new gradients
    optimizer.step()  # perform one gradient descent step

    return loss.item()

def train_dqn(
    env,
    num_steps,
    *,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    batch_size,
    exploration,
    gamma,
):
    """
    DQN algorithm.

    Compared to previous training procedures, we will train for a given number
    of time-steps rather than a given number of episodes.  The number of
    time-steps will be in the range of millions, which still results in many
    episodes being executed.

    Args:
        - env: The openai Gym environment
        - num_steps: Total number of steps to be used for training
        - num_saves: How many models to save to analyze the training progress.
        - replay_size: Maximum size of the ReplayMemory
        - replay_prepopulate_steps: Number of steps with which to prepopulate
                                    the memory
        - batch_size: Number of experiences in a batch
        - exploration: a ExponentialSchedule
        - gamma: The discount factor

    Returns: (saved_models, returns)
        - saved_models: Dictionary whose values are trained DQN models
        - returns: Numpy array containing the return of each training episode
        - lengths: Numpy array containing the length of each training episode
        - losses: Numpy array containing the loss of each training batch
    """

    # get the state_size from the environment
    state_size = env.Preprocess_stack(env.observations()).shape
    # print(state_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # initialize the DQN and DQN-target models
    dqn_model = DQN(state_size, env.numActions()).to(device)
    dqn_target = DQN.custom_load(dqn_model.custom_dump()).to(device)

    # initialize the optimizer
    optimizer = torch.optim.Adam(dqn_model.parameters())

    # initialize the replay memory and prepopulate it
    memory = ReplayMemory(replay_size, state_size)
    memory.populate(env, replay_prepopulate_steps)

    # initiate lists to store returns, lengths and losses
    rewards = []
    returns = []
    lengths = []
    losses = []

    # initiate structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # use this to indicate the index of the current episode
    t_episode = 0  # use this to indicate the time-step inside current episode

    env.reset()  # initialize state of first episode
    state = env.Preprocess_stack(env.observations())
    # print(state)

    # iterate for a total of `num_steps` steps
    pbar = trange(num_steps, desc="Num_steps", leave=False)
    for t_total in pbar:
        # use t_total to indicate the time-step from the beginning of training

        step = 0
        done = False
        total_rewards = 0
        # env.print_step(env.observations())
        # save model
        if t_total in t_saves:
            model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)

        # YOUR CODE HERE:
        #  * sample an action from the DQN using epsilon-greedy
        #  * use the action to advance the environment by one step
        #  * store the transition into the replay memory

        state = torch.tensor(state, dtype=torch.float).to(device)
        
        eps = exploration.value(t_total)
        prob = np.random.random()
        
        if prob > eps:
            state = state.unsqueeze(0)
            # print("s", state.shape)
            Q = dqn_model(state).detach()
            Q_cpu = Q.cpu()
            action = np.argmax(Q_cpu.numpy())
        else:
            action = np.random.choice(env.numActions())
            
        reward, done = env.step(action)
        # done = not env.is_running()

        if done:
            # YOUR CODE HERE:  anything you need to do at the end of an
            # episode, e.g. compute return G, store stuff, reset variables,
            # indices, lists, etc.

            # env.reset()
            # next_state = env.Preprocess_stack(env.observations())
            next_state = np.zeros(state_size)
            memory.add(state, action, reward, next_state, done)
            rewards.append(reward)
            lengths.append(len(rewards))
            
            G = 0
            for r in rewards:
                G = r + gamma * G
                
            if G > 15:
                G = returns[-1]
            returns.append(G)

            rewards = []

            pbar.set_description(
                f'Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}'
            )
            
            i_episode += 1
            t_episode = 0
            env.reset()
            state = env.Preprocess_stack(env.observations())
            
        else:
            # YOUR CODE HERE:  anything you need to do within an episode
            next_state = env.Preprocess_stack(env.observations())
            memory.add(state, action, reward, next_state, done)
            rewards.append(reward)
            t_episode += 1
            state = next_state
        
        # YOUR CODE HERE:  once every 4 steps,
        #  * sample a batch from the replay memory
        #  * perform a batch update (use the train_dqn_batch() method!)
        
        if t_total % 4 == 3:
            batch_sampled = memory.sample(batch_size)
            
            loss = train_dqn_batch(optimizer, batch_sampled, dqn_model, dqn_target, gamma)
            losses.append(loss)

        # YOUR CODE HERE:  once every 10_000 steps,
        #  * update the target network (use the dqn_model.state_dict() and
        #    dqn_target.load_state_dict() methods!)
        
        if t_total % 10000 == (10000-1):
            dqn_target.load_state_dict(dqn_model.state_dict())

    saved_models['100_0'] = copy.deepcopy(dqn_model)

    return (
        saved_models,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )

def Test(model_name, num_steps, test_display = True):
    try:
        checkpoint = torch.load(model_name)
    except FileNotFoundError:
        pass
    else:
        value = checkpoint['100_0']
        dqn = DQN.custom_load(value)

    reward_total = 0
    is_terminal = False
    rewards = []
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # num_episodes = 30

    pbar = trange(num_steps, desc="Num_steps", leave=False)
    for t_total in pbar:
    # while (num_steps != 0):
        # num_steps -= 1
        if (is_terminal):
            env.reset()
            is_terminal = False
            pbar.set_description(
                f'Return: {reward_total:5.2f}'
            )
            if reward_total > 15:
                reward_total = rewards[-1]
            rewards.append(reward_total)
            # print("Total reward: {}".format(reward_total))
            reward_total = 0

        state_raw = env.observations()
        # env.print_step(env.observations())

        state = env.Preprocess_stack(state_raw)
        state = torch.tensor(state, dtype=torch.float)#.to(device)
        # action = dqn(state)
        state = state.unsqueeze(0)
        Q = dqn(state).detach()
        Q_cpu = Q.cpu()
        action = np.argmax(Q_cpu.numpy())

        for _ in range(4):
            # Display.
            if (test_display):
                cv2.imshow("frame-test", state_raw)
                cv2.waitKey(20)

            # if (test_write_video):
            #     out_video.write(state_raw)

            reward, is_terminal = env.step(action, 1)
            reward_total += reward

            if (not env.is_running()):
                break

            state_raw = env.observations()
        # print(reward_total)
    return np.array(rewards)

from env_lab import EnvLab
import six

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
    parser.add_argument('--model', type=str, default='/home/pengk/git_ws/lab/models/checkpoint_default.pt',
                        help='The name of the model')
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='The number of steps to play.')
    
    parser.add_argument('--Test', type=bool, default=False,
                        help='Test Mode')
    
    args = parser.parse_args()

  # Convert list of level setting strings (of the form "key=value") into a
  # `config` key/value dictionary.
    config = {
        k: v
        for k, v in [six.ensure_str(s).split('=') for s in args.level_settings]
    }

    channel = 1

    env = EnvLab(args.level_script, 84, 84, channel, config)
    gamma = 1
    # Create an action to move forwards.
    num_steps = args.num_steps
    num_saves = 11 # if it is 5, then save models at 0%, 25%, 50%, 75% and 100% of training

    replay_size = 100_000
    replay_prepopulate_steps = 1_500  # 50_000

    batch_size = 64
    exploration = ExponentialSchedule(1.0, 0.01, num_steps * 0.8)

    if args.Test:
        returns = Test(args.model, num_steps)

        plt.figure(2)
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        plt.title('DQN 4 Image Queue')
        plt.plot(returns, color = 'pink', label = 'Returns')
        plt.plot(range(len(returns)), [np.average(returns)] * len(returns), 'k', linestyle='dashed', label = 'Average')
        # plt.plot(rolling_average(returns, window_size = int(len(returns)/200)),'r', label = 'Average')
        plt.legend()
    else:
        # this should take about 90-120 minutes on a generic 4-core laptop
        dqn_models, returns, lengths, losses = train_dqn(
            env,
            num_steps,
            num_saves=num_saves,
            replay_size=replay_size,
            replay_prepopulate_steps=replay_prepopulate_steps,
            batch_size=batch_size,
            exploration=exploration,
            gamma=gamma,
        )

        checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
        torch.save(checkpoint, args.model)

        plt.figure(1)
        plt.xlabel('Episodes')
        plt.ylabel('Losses')
        plt.title('DQN 4 Image Queue')
        plt.plot(losses, 'pink', label = 'Losses')
        plt.plot(rolling_average(losses, window_size = int(len(losses)/200)),'r', label = 'Average')
        plt.legend()

        ### YOUR PLOTTING CODE HERE

        plt.figure(2)
        plt.xlabel('Episodes')
        plt.ylabel('Return')
        plt.title('DQN 4 Image Queue')
        plt.plot(returns, color = 'pink', label = 'Returns')
        plt.plot(rolling_average(returns, window_size = int(len(returns)/200)),'r', label = 'Average')
        plt.legend()

    # plt.figure(2)
    # plt.xlabel('Episodes')
    # plt.ylabel('Length')
    # plt.title('Mountaincar')
    # plt.plot(lengths, 'pink', label = 'Lengths')
    # plt.plot(rolling_average(lengths, window_size = int(len(lengths)/200)),'r', label = 'Average')
    # plt.legend()


    plt.show()
        # for _ in six.moves.range(args.num_episodes):
        #     while env.is_running():
        #         # Advance the environment 4 frames while executing the action.
        #         reward = env.step(action)

        #         if reward != 0:
        #             score += reward
        #             obs = env.observations()  # dict of Numpy arrays
        #             # rgb_i = obs['RGB_INTERLEAVED']
        #             print('Observation shape:', obs.shape)