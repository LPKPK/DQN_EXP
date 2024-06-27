# DQN Experiment
## Introduction
The experiment is to find out the ways to improve simple DQN performance.


## Installation
1. Install [DeepMind Lab](https://github.com/id-Software/Quake-III-Arena](https://github.com/deepmind/lab))
2. Replace `BUILD` in DeepMind Lab repo, ensure it has the correct path to srcs
3. Copy agent_*.py scripts to lab/python/
4. Make the necessary adjustments to the `.lua` file to configure a different map.

## Train a Model
The default path to save a model is 'models/'
```shell
gym$ bazel run :DQN_imgstk -- --level_script=/home/pengk/git_ws/lab/game_scripts/levels/seekavoid_arena_01.lua   --num_steps 1000000
```

## Run a Pre-trained Model
```shell
gym$ bazel run :DQN_imgstk -- --level_script=/home/pengk/git_ws/lab/game_scripts/levels/seekavoid_arena_01.lua   --num_steps 3050 --Test True   --model  /home/pengk/git_ws/lab/models/lenet_4img_1000000.pt
```

## Result
Reward: 

Apple: +1  |  Lemon: -1


![RL_deepmind](https://github.com/LPKPK/DQN_EXP/assets/51788243/242d8698-6499-4a58-b2ad-ea048c78f0f9)
