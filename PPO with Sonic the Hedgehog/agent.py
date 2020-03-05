import tensorflow as tf
import numpy as np
import gym
import math
import os
import time

import model
import architecture as policies
import sonic_env as env

# SubprocVecEnv creates a vector of n environments to run them simultaneously.
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

def main():
    t_start = time.time()
    tf.logging.set_verbosity(tf.logging.ERROR)
    config = tf.ConfigProto()
    
    # Avoid warning message errors
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    # Allowing GPU memory growth
    config.gpu_options.allow_growth = True
    
    # env = DummyVecEnv([env.make_train_4])
    # env = SubprocVecEnv([env.make_train_4])
     
                                                # env.make_train_1, 
                                                # env.make_train_2, 
                                                # env.make_train_3, 
                                                # env.make_train_4, 
                                                # env.make_train_5,
                                                # env.make_train_6,
                                                # env.make_train_7,
                                                # env.make_train_8,
                                                # env.make_train_9,
                                                # env.make_train_10,
                                                # env.make_train_11,
                                                # env.make_train_12]), 
    
    with tf.Session(config=config):
        model.learn(policy=policies.PPOPolicy,
                            # env = DummyVecEnv([env.make_train_4]),
                            env = SubprocVecEnv([env.make_train_0,
                                                 env.make_train_1,
                                                 env.make_train_2,
                                                 env.make_train_3,
                                                 env.make_train_4,
                                                 env.make_train_5,
                                                 env.make_train_6,
                                                 env.make_train_7,
                                                 env.make_train_8,
                                                 env.make_train_9,
                                                 env.make_train_10,
                                                 env.make_train_11,
                                                 env.make_train_12,
                                                 env.make_train_13,
                                                 env.make_train_14,
                                                 env.make_train_15,
                                                 env.make_train_16,
                                                 env.make_train_17]),
                                                 # env.make_train_18]),
                            nsteps=2048, # Steps per environment
                            total_timesteps=1474560,
                            #total_timesteps=10000000,
                            gamma=0.99,
                            lam = 0.95,
                            vf_coef=0.5,
                            ent_coef=0.01,
                            lr = lambda _: 2e-4,
                            cliprange = lambda _: 0.1, # 0.1 * learning_rate
                            max_grad_norm = 0.5, 
                            log_interval = 10
                            )
    
    t_end = time.time()
    print("Time elapsed", time.strftime('%H:%M:%S', t_end - t_start))
    
if __name__ == '__main__':
    main()


