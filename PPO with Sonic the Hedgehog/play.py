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
    
    env = DummyVecEnv([env.make_train_4])
    
    actions = model.play(policy=policies.PPOPolicy, 
                         env=env, 
                         update=1)

env.close()

# from matplotlib import pyplot as plt
# if actions != -1:
#     plt.plot(actions)
#     plt.show()
    
#     unique, counts = np.unique(actions, return_counts=True)
#     action_groups = dict(zip(unique, counts))
#     plt.bar(range(len(action_groups)), list(action_groups.values()), align='center')
#     plt.xticks(np.arange(7), ['LEFT', 'RIGHT', 'LEFT+DOWN', 'RIGHT+DOWN', 'DOWN',
#                        'DOWN+B', 'B'], rotation=45)
#     plt.gca().set_ylabel('Count')
#     plt.title('Model 500 - First 1000 actions')











