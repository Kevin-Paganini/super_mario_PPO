import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from train import SuperMarioGym
import os

from ray.tune.registry import get_trainable_cls
# Initialize Ray
ray.init()

config = (
        get_trainable_cls("PPO")
        .get_default_config()
        .environment(SuperMarioGym)
        .framework("torch")
        .training(
            model={
            "dim": (20, 16),
            
            "conv_filters": [[1024, [4, 4], 1], [1024, [4, 4], 1]],
            
            
            },
            train_batch_size=10000000, 
            num_sgd_iter=2048, 
            sgd_minibatch_size=1024,
            vf_clip_param=100,
            lr=1e-8)
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .rl_module(_enable_rl_module_api=False)  # Deactivate RLModule API
        .training(_enable_learner_api=False)  # Deactivate RLModule API
        
        
    )
ch_path = os.path.join(os.getcwd(), 'trained', 'test_19_exp_decay_2000')
agent = config.build()
agent.restore(ch_path)
agent.config['num_rollout_workers'] = 1
print(agent.config)
print(f"Agent loaded from saved model at {ch_path}")

# inference
env = SuperMarioGym({})
obs, info = env.reset()
terminated = False
truncated = False
while not terminated or not truncated:
    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    