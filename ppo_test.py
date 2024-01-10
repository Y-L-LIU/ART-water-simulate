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
    max_ep_len = env_1.ep_length           # max timesteps in one episode
    print(max_ep_len)
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
        print(state.shape)
        for t in range(1, max_ep_len+1):
            s = []
            tem_before = state[-5]
            action = ppo_agent.select_action(state)
            s.append(tem_before + action[0])
            s.append(action)
            s.append(state[-5])
            test_result.append(s)
            state, reward, done, _ = env.step(action)
            
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break


        # 文件名
        file_name = 'output.xlsx'
        # print(test_result[0])
        # 将二维数组写入 CSV 文件
        import pandas as pd
        # data_col real
        data_col = pd.read_excel('./data/test.xlsx')['date']
        #data_col_1h
        data1 = pd.read_excel('./data/test.xlsx', sheet_name='Sheet1')
        data1['date'] = pd.to_datetime(data1['date'])
        data2 = pd.read_excel('./data/test.xlsx', sheet_name='Sheet2')
        data2['date'] = pd.to_datetime(data2['date'])
        merged_data = pd.merge_asof(data2, data1, on='date', tolerance=pd.Timedelta('1H'))
        data_col_1h = merged_data['date']
        out = pd.DataFrame(test_result)
        out['date'] = data_col_1h
        # merge back
        df_new = pd.merge_asof(data_col, out, on='date', tolerance=pd.Timedelta('1H'))
        df_new.to_excel(file_name)
        df_new.to_csv('output.csv')
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