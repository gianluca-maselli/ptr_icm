import torch
from IntrinsicCuriosityModule.replay_buffer import ReplayBuffer
from IntrinsicCuriosityModule.model import DQN_ICM
from IntrinsicCuriosityModule.icm import ICM
from IntrinsicCuriosityModule.train_icm_only import train_icm
from IntrinsicCuriosityModule.test_icm import test_icm
import imageio
from utils import *
from torch.utils.data import DataLoader
import copy
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
#there are two sets of action-spaces we can import: one with 5 actions (simple) and one with 12 (complex).
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from PIL import Image 
import random
import numpy as np
import os
import shutil
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.insert(0, './TRNET')

start_time = time.time()

if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device', device)
    SEED= 90
    # set global seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if device == 'cuda':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(SEED)
    
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    #Wraps the environment’s action space to be 12 discrete actions
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env.seed(SEED)
    print(COMPLEX_MOVEMENT)
    env.reset()
    
    # --------- ICM ---------#
    
    # icm_dqn parameters
    layers_dim = [32,32,32, 32]
    kernel_sizes = [3,3,3,3]
    strides = [2,2,2,2] 
    paddings = [1] 
    fc1_dim = 256

    #commond parameters
    input_shape = 4 #n_channels
    out_dim = env.action_space.n

    # ptr_dqn parameters
    
    dqn_icm = DQN_ICM(input_shape=input_shape, layers=layers_dim, kernel_sizes=kernel_sizes, strides=strides, fc_dim=fc1_dim, out_dim=out_dim, padding=paddings, device=device).to(device)

    dqn_icm_target = copy.deepcopy(dqn_icm)
    dqn_icm_target.load_state_dict(dqn_icm.state_dict())
    
    dqn_icm_loss = torch.nn.MSELoss()

    icm_model = ICM(input_shape=input_shape, layers=layers_dim, kernel_sizes=kernel_sizes, strides=strides, fc_dim=fc1_dim, out_dim=out_dim, padding=paddings, device=device).to(device)

    optimizer = torch.optim.Adam(list(dqn_icm.parameters()) + list(icm_model.parameters()), lr=1e-4)

    dqn_icm.train()
    icm_model.train()
    
    buffer_size = 100000
    buffer_icm = ReplayBuffer(state_size=(buffer_size, 4, 42, 42), action_size=1, buffer_size=buffer_size, device=device)
    
    params = {
        'batch_size':128,
        'beta':0.2,
        'lambda': 0.95, #0.1,
        'eta': 1.0, #(x)
        'gamma': 0.99, #(x)
        'action_repeats_icm':6,
        }
    
    # -------- TRAIN ICM -------------#
    train_icm(env, dqn_icm, dqn_icm_target, icm_model, optimizer, buffer_icm, params, device)

