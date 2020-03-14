import tensorflow as tf
import numpy as np
import gym
import math
import os

#import retro_contest

import model
import architecture as policies
import sonic_env as env

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

#def main():
tf.logging.set_verbosity(tf.logging.ERROR)
config = tf.ConfigProto()

# Avoid warning message errors
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Allowing GPU memory growth
config.gpu_options.allow_growth = True

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.Session(config=config):
    
    env = DummyVecEnv([env.make_test])
    update = 2640
    
    play_result = model.play(policy=policies.PPOPolicy, 
                             env=env, 
                             update=update)
    
    actions, rewards, score, info = play_result

env.close()

# The Simonini code scales the reward by 0.01 to work better
# with PPO.

flat_rewards = [item for sublist in rewards for item in sublist]
#flat_rewards = np.array(flat_rewards)*100
cum_rewards = np.cumsum(flat_rewards)

# from matplotlib import pyplot as plt
# if actions != -1:
#     plt.plot(actions)
#     plt.show()
    
#     unique, counts = np.unique(actions, return_counts=True)
#     action_groups = dict(zip(unique, counts))
#     plt.bar(range(len(action_groups)), list(action_groups.values()), align='center')
#     plt.xticks(np.arange(7), ['LEFT', 'RIGHT', 'LEFT+DOWN', 'RIGHT+DOWN', 'DOWN',
#                         'DOWN+B', 'B'], rotation=45)
#     plt.gca().set_ylabel('Count')
#     plt.title('Model ' + str(update) + ' - First 1000 actions')

from matplotlib import pyplot as plt
plt.plot(rewards, linewidth=0.75 , color='orange')
plt.gca().set_ylabel('Reward')
plt.gca().set_xlabel('Step')
plt.title('Model ' + str(update) + ' - Score: ' + str(score[0]))

ax2 = plt.gca().twinx() 
plt.plot(cum_rewards, label='Cumulative', linewidth=0.75)
plt.gca().set_ylabel('Cumulative sum of Rewards')
ax2.legend(loc=7)

plt.show()

actions[5]
info[3000]
flat_rewards[5]

offset_x = info[0][0]['offset_x']


x_progress = [info[i][0]['x'] + offset_x  for i in range(len(info))]
x_diff = [info[i][0]['x'] - info[i][0]['screen_x']  for i in range(len(info))]

plt.plot(x_diff)
plt.gca().set_ylabel('x Progress')
plt.gca().set_xlabel('Step')
plt.title('Model ' + str(update) + ' - x progress: ' + str(score[0]))
plt.show()




