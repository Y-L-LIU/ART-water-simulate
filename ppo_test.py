import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import csv
import gym

from src.ppo import PPO


#################################### Testing ###################################
def test(env_1):
    print("============================================================================================")
    env_name = "water"
    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 1    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = env_1

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    test_result = []
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            s = []
            tem_before = state[-5]
            action = ppo_agent.select_action(state)
            
            state_pred = state.reshape((64,8))
            state_pred[-1,3] = state_pred[-1,3]+action
            tem_in = env.get_tem(state_pred)
            s.append(tem_in)
            s.append(float(tem_before) + float(action))
            test_result.append(s)
            state, reward, done, _ = env.step(action)
            
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break


        # 文件名
        file_name = 'output.csv'
        # print(test_result[0])
        # 将二维数组写入 CSV 文件
        with open(file_name, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for row in test_result:
                csv_writer.writerow(row)

        print(f"已将二维数组写入 {file_name}")

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    from src.env import IdentityEnv
    path_to_model = '/home/yuleliu/art/model_weight/model_step64_dropout_0.1_dim_256_layer_16_test_0.274868298345625.pth'
    env = IdentityEnv(64,path_to_model,True)
    test(env)