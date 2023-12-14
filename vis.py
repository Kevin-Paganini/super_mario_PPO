import matplotlib.pyplot as plt
import numpy as np


with open('vis_stats.txt', 'r') as f:
    lines = f.read().strip().split('\n')
    
ret = []
for i, l in enumerate(lines):
    temp = l.split("episode_reward_mean\': ")[-1]
    ep_reward_mean = float(temp.split(',')[0])
    ret.append((i*5, ep_reward_mean))
    
ret = np.array(ret)
print(ret)

fig, ax = plt.subplots(figsize=(15, 10))

ax.scatter(ret[:, 0], ret[:, 1])

ax.set_xlabel('Episode Number', fontsize=14)
ax.set_ylabel('Episode Reward Mean', fontsize=14)
ax.set_title('Episode Reward Mean Over Time', fontsize=16)

plt.savefig('ep_progress')
    



