#!/usr/bin/env python3

import argparse
from email.policy import default
from traceback import print_tb
import gym
from pyparsing import replaceWith

import ray
from ray import tune
from importlib.resources import path
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.logger import pretty_print
from envs.vacuum_cleaner_env import VacuumCleanerEnv

parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=4000)
parser.add_argument("--checkpoint-path", type=str, default=None)
parser.add_argument("--use-safe-env", type=bool, default=False)

config = {
    "env": VacuumCleanerEnv,
    "env_config": {
        "desc": None,
        "map_name":"6x6",
        "is_slippery":False,
    },
    "explore": True,
    "exploration_config": {
        "type": "EpsilonGreedy",
        # Parameters for the Exploration class' constructor:
        # "initial_epsilon"=1.0,  # default is 1.0
        # "final_epsilon"=0.05,  # default is 0.05
        "epsilon_timesteps": 1e7,  # Timesteps over which to anneal epsilon, defult is int(1e5).
    },
    # "horizon": 500,
    "num_workers": 18,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [64, 64],    # Number of hidden layers
        "fcnet_activation": "relu", # Activation function
    },
    "evaluation_num_workers": 4,
    "evaluation_interval": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": False,
    },
}

class MyExperiment():
    def __init__(self):
        ray.shutdown()
        ray.init()
        self.config = config
        self.trainer = PPOTrainer
        self.env_class = config["env"]

    def train(self, stop_criteria):
        results = tune.run(
            self.trainer,
            stop=stop_criteria,
            config=self.config,
            local_dir="./results",
            keep_checkpoints_num=3,
            checkpoint_freq=5,
            checkpoint_at_end=True,
            verbose=1,
        )

        results.default_mode = "max"
        checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean'),
                                                          metric="episode_reward_mean")
        
        checkpoint_path = checkpoints[-1][0]
        print("Checkpoint path:", checkpoint_path)
        return checkpoint_path, results
    
    def load(self,path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = PPOTrainer(config=self.config)
        self.agent.restore(path)

    def test(self):
        """
        Test a trained agent for an episode. Return the episode reward.
        """
        
        env = self.env_class(env_config=self.config["env_config"])
        
        episode_reward = 0
        done = False
        obs = env.reset()
        
        while not done:
            env.render()
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
        
        return episode_reward
        
def main(stop_criteria=None, use_safe_env=False, checkpoint_path=None):
    exp = MyExperiment()
    
    if (checkpoint_path == None):
        print("No checkpoint provided - training from scratch")
        checkpoint_path, results = exp.train(stop_criteria={"training_iteration": stop_criteria})
        print("Finished training!")
    else :
        print("Checkpoint provided - loading from checkpoint")    
    
    exp.load(checkpoint_path)
    print("Finished loading!")
    
    # checkpoint_path = "/home/matteo/ray_results/PPO_2022-04-22_15-59-50/PPO_VacuumCleanerEnv_7a549_00000_0_2022-04-22_15-59-50/checkpoint_004000/checkpoint-4000"
    
    print("Testing trained agent!\nLoading checkpoint:", checkpoint_path)    
    exp.load(checkpoint_path)
    
    print("Starting testing")
    #TODO: visualize environment
    r= exp.test()
    print("Finished testing! Cumulative Episode Reward:",r)
    
if __name__ == "__main__":
    args = parser.parse_args()
    use_safe_env = args.use_safe_env
    checkpoint = args.checkpoint_path
    main(args.stop_iters, use_safe_env, checkpoint)
