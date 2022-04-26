# WASP_AI_Course2022

Repo for assignments of the course WASP Artificial Intelligence and Machine Learning 2022.

The repo contains the code to run a simple agent that simulate a vacuum cleaner that needs to clean a room full of people.
The agent is trained to clean the room by using a reinforcement learning algorithm in particular the [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) algorithm.

The code is divided in the environment which rapresents the room and the agent interacting with the environment.

## Installation

How to install the libraries. 
The code use the [ray\[rllib\]](https://docs.ray.io/en/latest/rllib/index.html) library to train the agent.

To install the library using anaconda you need to install the following libraries:

```bash
conda create -n rllib python=3.8
conda activate rllib
pip install "ray[rllib]" tensorflow torch
```

## Run the code

To run the code you can both used a pre-trained agent or train an agent from scratch.

To use the pretrain agent you can use the checkpoints saved in the checkpoints folder:

```bash
    python3 vacuum_cleaner_exp.py --checkpoint <PATH_TO_REPO>/WASP_AI_Course2022/checkpoints/PPOTrainer_2022-04-25_16-42-57/PPOTrainer_VacuumCleanerEnv_ffac3_00000_0_2022-04-25_16-42-58/checkpoint_000180/checkpoint-180
```

To run a training from scratch you can just run:
```bash
    python3 vacuum_cleaner_exp.py
```
