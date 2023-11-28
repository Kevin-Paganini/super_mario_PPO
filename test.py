import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from train import SuperMarioGym
import os
# Initialize Ray
ray.init()

# load and restore model
agent = ppo.PPO(env=SuperMarioGym)
ch_path = os.path.join(os.getcwd(), 'trained', 'test_3_40')
agent.restore(ch_path)
print(f"Agent loaded from saved model at {ch_path}")

# inference
env = gym.make(SuperMarioGym)
obs, info = env.reset()
for i in range(1000):
    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        print(f"Cart pole ended after {i} steps.")
        break
    