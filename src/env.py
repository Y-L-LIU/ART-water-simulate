import torch
from typing import Any, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
from openrl.envs.common import make
from src.mlpmixer import MLPMixer
import json

def load_data(time_steps,excel_file = 'data/train.xlsx'):
    import pandas as pd
    excel_file = 'data/train.xlsx'
    data1 = pd.read_excel(excel_file, sheet_name='Sheet1')
    data1['date'] = pd.to_datetime(data1['date'])
    data2 = pd.read_excel(excel_file, sheet_name='Sheet2')
    data2['date'] = pd.to_datetime(data2['date'])
    merged_data = pd.merge_asof(data2, data1, on='date', tolerance=pd.Timedelta('1H'))
    merged_data.keys()
    selected_columns = ['pri_supp_t','pri_back_t', 'pri_flow', 'sec_supp_t', 'sec_back_t', 'sec_flow', 'outdoor', 'irradiance']
    data_subset = merged_data[selected_columns]
    data_subset.dropna(inplace=True)
    X = data_subset.values
    X_sequence = []
    tmp = []
    for i in range(len(data_subset) - time_steps):
        tmp.append(X[i:i+time_steps])
        X_sequence.append((np.concatenate(X[i:i+time_steps])))
    X_sequence = np.array(X_sequence, dtype=np.float32)
    return X_sequence, np.array(tmp, dtype=np.float32 )

def yield_data(data):
    for x in data:
        yield x

def get_seq(excel_file):
    import pandas as pd
    data1 = pd.read_excel(excel_file, sheet_name='Sheet1')
    data1['date'] = pd.to_datetime(data1['date'])
    data2 = pd.read_excel(excel_file, sheet_name='Sheet2')
    data2['date'] = pd.to_datetime(data2['date'])
    merged_data = pd.merge_asof(data2, data1, on='date', tolerance=pd.Timedelta('1H'))
    merged_data.keys()
    selected_columns = ['pri_supp_t','pri_back_t', 'pri_flow', 'sec_supp_t', 'sec_back_t', 'sec_flow', 'outdoor', 'irradiance']
    data_subset = merged_data[selected_columns]
    data_subset.dropna(inplace=True)
    X = data_subset.values
    return X

def load_test_data(time_steps):
    X1, X2 = get_seq('data/train.xlsx'), get_seq('data/test.xlsx')
    print(X1.shape, X2.shape)
    X = np.concatenate([X1[-64:], X2], axis=0)
    X_sequence = []
    tmp = []
    for i in range(len(X) - time_steps):
        tmp.append(X[i:i+time_steps])
        X_sequence.append((np.concatenate(X[i:i+time_steps])))
    X_sequence = np.array(X_sequence, dtype=np.float32)
    return X_sequence, np.array(tmp, dtype=np.float32 )

def load_config(path):
    return json.load(path, 'r')

class IdentityEnv(gym.Env):
    spec = EnvSpec("IdentityEnv")
    def __init__(self,time_steps,path_to_model,is_test=False):
        self.dim = 2
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.dim)
        self.time_steps = time_steps
        self.current_step = 0
        self.is_test = is_test
        if not is_test:
            datas, tmp = load_data(time_steps)
        else:
            datas, tmp = load_test_data(time_steps)
        self.ep_length = len(datas)
        self.yield_data = yield_data(datas)
        self.yield_predict = yield_data(tmp)
        #config for openrl
        self.agent_num=1
        self.parallel_env_num = 1      
        self.env_name = 'water'  
        # Get action space
        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,))
        # Get observation space
        max_value = []
        min_value = []
        for i in range(8):
            max_value.append(round(tmp[:,:,i].max())+1)
            min_value.append(round(tmp[:,:,i].min())-1)
        self.observation_space = gym.spaces.Box(low=np.array(min_value*time_steps), high=np.array(max_value*time_steps),shape=(time_steps*8,))
        #import the simulator
        config = json.load(open('/home/yuleliu/art/model_weight/model_config.json','r'))
        self.model = MLPMixer(**config,batched=False)
        self.model.to(self.model.device)
        self.model.load_state_dict(torch.load(path_to_model))

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
    ):
        if seed is not None:
            self.seed(seed)
        self.current_step = 0
        if not self.is_test:
            datas, tmp = load_data(self.time_steps)
        else:
            datas, tmp = load_test_data(self.time_steps)
        self.yield_data = yield_data(datas)
        self.yield_predict = yield_data(tmp)
        return next(self.yield_data)
    

    def get_tem(self, x):
        with torch.no_grad():
            self.model.eval()
            predicted = self.model(torch.from_numpy(x))
        return float(predicted)

    def step(self, action) :
        try:
            state = next(self.yield_data)
        except:
            state = None
        state_pred = next(self.yield_predict)
        tem_origin = state_pred[-1,3]
        # print(state_pred[-1,3])
        state_pred[-1,3] = state_pred[-1,3]+action
        tem_in = self.get_tem(state_pred)
        reward = 0
        if 20<=tem_in<=24:
            reward+=1
            if state_pred[-1,3]<tem_origin:
                reward+=0.5
            else:
                reward-=0.5
        else:
            reward-=1
        self.current_step += 1
        done = self.current_step >= self.ep_length
        return state, reward, done, {}
    
    def render(self, mode: str = "human") -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            
    def close(self):
        pass
    
