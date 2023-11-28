from ray.rllib.algorithms.ppo import PPOConfig


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Example of a using a custom gym environment with RLlib.
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random
import sys

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


from models import PolicyNetwork

from pyboy import PyBoy, WindowEvent # isort:skip
torch, nn = try_import_torch()

ACTIONS = [
    
    [WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_ARROW_LEFT],
    [WindowEvent.PRESS_BUTTON_A],
    [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_RIGHT],
    [WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT],
    [WindowEvent.RELEASE_ARROW_RIGHT],
    [WindowEvent.RELEASE_ARROW_LEFT]
]

NUM_WORKERS = os.cpu_count() - 5
print(os.cpu_count())
TICK_RANGE = 5

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#Define a class that specifies a corridor environment
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class SuperMarioGym(gym.Env):
    """Super Mario Gym"""

    def __init__(self, config: EnvContext):
        

        

        # Check if the ROM is given through argv
        filename = 'super_mario.gb'

        quiet = "--quiet" in sys.argv
        self.pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=not quiet, game_wrapper=True)
        self.pyboy.set_emulation_speed(0)
        assert self.pyboy.cartridge_title() == "SUPER MARIOLAN"

        self.mario = self.pyboy.game_wrapper()
        self.mario.start_game() 
        
        
        
        self.previous_progress = 0
        
        self.action_space = Discrete(len(ACTIONS))    #set the nature of the action space
        self.observation_space = Box(0.0, 500, shape=(20,16), dtype=np.float32)    #the state space -- just the position in the corridor
        self.reset(seed=8)    # Set the seed. This is only used for the final (reach goal) reward.



    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.mario.reset_game()
        assert self.mario.score == 0
        assert self.mario.lives_left == 2
        assert self.mario.time_left == 400
        assert self.mario.world == (1, 1)
        self.previous_progress = 0

        self.last_lives_left = self.mario.lives_left
        
        
        return np.array(self.mario.game_area(), dtype=np.float32).reshape((20, 16)), {}

    def step(self, action):
        print(action)
        done = False
        str_action = ACTIONS[action]
        if len(str_action) > 1:
            
            self.pyboy.send_input(str_action[0])
            self.pyboy.send_input(str_action[1])
        else:
            self.pyboy.send_input(str_action[0])
        
        for i in range(TICK_RANGE):
            if self.mario.lives_left <= 0:
                done = True
            else:
                self.last_lives_left = self.mario.lives_left
            self.pyboy.tick()
        
        if WindowEvent.PRESS_BUTTON_A in str_action:
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
        
        # Need something to get it to actually move to the right
        reward = self.mario.level_progress - 251 # where mario starts
        if self.mario.level_progress <= self.previous_progress:
            reward = reward * 0.98
        else:
            reward = reward * 1.02
        
        self.previous_progress = self.mario.level_progress
        
        return (
            np.array(self.mario.game_area(), dtype=np.float32).reshape((20, 16)),    #the current state
            reward,    #generate a random reward if done otherwise accumulate -0.1
            done,    #record whether the agent has reached the end point
            False,   #""
            {},    #
        )
        
 
        
def train():
    #@@@@@@@@@@@@@@@@@@@
    #Train the algorithm
    #@@@@@@@@@@@@@@@@@@@
    
    
    file_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(0, file_path + "/..")
    
    ray.init()    #spin up distributed computing using ray

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment(SuperMarioGym)
        .framework("torch")
        .training(
            model={
            "dim": (16, 20),
            
            "conv_filters": [[1024, [4, 4], 1], [1024, [4, 4], 1]],
            
            },
            train_batch_size=4096, 
            num_sgd_iter=2048, 
            sgd_minibatch_size=1024,
            vf_clip_param=100,
            lr=1e-2)
        .rollouts(num_rollout_workers=NUM_WORKERS)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rl_module(_enable_rl_module_api=False)  # Deactivate RLModule API
        .training(_enable_learner_api=False)  # Deactivate RLModule API
        
        
    )

    stop = {
        "training_iteration": 1000,    #set the number of training iterations
        "timesteps_total": 10,    #set the total number of timesteps
        "episode_reward_mean": 0.1    #set the average reward that will stop training once achieved
    }
    
    algo = config.build()    #build the algorithm using the config
    # Create a PPOTrainer and load the saved model

    # algo.restore(os.path.join(os.getcwd(), 'trained', 'test_3_40'))
    # print(dir(algo))
    # algo.config['num_rollout_workers'] = 1
    
    for iteration in range(stop['training_iteration']):    #loop over training iterations
        result = algo.train()    #take a training step
        print('The agent encountered',result['episodes_this_iter'],'episodes this iteration...')
        if iteration % 5 == 0:
            os.makedirs(os.path.join(os.getcwd(), 'trained', f'{iteration}'), exist_ok=True)
            checkpoint_path = algo.save(os.path.join(os.getcwd(), 'trained', f'{iteration}'))
            steps_trained = result['info']['learner']['default_policy']['num_agent_steps_trained']
            sample_results = result['sampler_results']
            print(f"Model saved at iteration {iteration}, steps trained: {steps_trained}, sample results: {sample_results}")
            with open(os.path.join(os.getcwd(), 'stats.txt'), 'a') as f:
                f.write(f"Model saved at iteration {iteration}, steps trained: {steps_trained}, sample results: {sample_results}\n")

    algo.stop()    #release the training resources
    ray.shutdown()    #and shut down ray
    
    
    # pyboy.stop()



if __name__ == "__main__":
    train()
