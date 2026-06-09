import torch
from IntrinsicCuriosityModule.replay_buffer import ReplayBuffer
from IntrinsicCuriosityModule.model import DQN_ICM
from IntrinsicCuriosityModule.icm import ICM
from IntrinsicCuriosityModule.train import train_icm
from PTR.utils import *
from PTR.model import NoisyDQN, NoisyLinear
from PTR.execute_path import path_finder
from PTR.recover_path import repeat_path
import imageio
from utils import *
import copy
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
#there are two sets of action-spaces we can import: one with 5 actions (simple) and one with 12 (complex).
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from similarity import GoalSimilarity
from PIL import Image 
import random
import numpy as np
import os
import shutil
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import time
import json
import itertools
import argparse
import statistics

parser = argparse.ArgumentParser(description='PTR')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

#selected cuda device where run the program, default 0
os.environ["CUDA_VISIBLE_DEVICES"]="0"

sys.path.insert(0, './TRNET')

start_time = time.time()


def filter_within_std(numbers):
    mean = statistics.mean(numbers)
    std_dev = statistics.stdev(numbers)

    lower_bound = mean - std_dev
    upper_bound = mean + std_dev

    filtered = [x for x in numbers if lower_bound <= x <= upper_bound]

    return filtered

if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device', device)
    args = parser.parse_args()
    SEED= args.seed
    print('SEED', SEED)
    # set global seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if device == 'cuda':
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(SEED)
    
    #create res directories
    dir_gif_goals = './gifs_goals'
    if os.path.exists(dir_gif_goals):
        shutil.rmtree(dir_gif_goals)
    os.makedirs(dir_gif_goals)

    
    #importing env
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    #Wraps the environment’s action space to be 12 discrete actions
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env.seed(SEED)
    #print(COMPLEX_MOVEMENT)
    env.reset()
    
    #saved models
    models_dict = {} #store {model_name_i: path_model_i}
    #save subgoals 
    sub_goals_dict = {} # store {model_name_i: x_goal_i}
    trjs__plots = {} #store {goal_i: test_trj_i} used to plot
    competences = {} #store {goal_i: list_C} used to plot
    # --------- ICM params ----------#
    
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
    
    icm_trials = 1 #10  #ICM total exploratory trials
    
    # -------------------------------#

    #training hyperparams (ICM and PTR)
    params = {
        'batch_size':128,
        'batch_size_ptr':32,
        'beta':0.2,
        'lambda': 0.95, #0.1,
        'eta': 1.0, #(x)
        'gamma': 0.99, #(x)
        'max_episode_len': 1500, #avoids Mario get stuck
        'action_repeats_icm':6,
        'action_repeats_ptr':4,
        }
    
    # -------------------------------#
    
    # ------ Goal selection functions -----#
    previous_goal  = 0
    #set goal similarity class
    goal_sim = GoalSimilarity()

    count_model = 0
    x_position = 0
    flag_get = False
    
    trj_n = 0
    
    # ----------PTR params -----------# 
    use_noisy = True
    lr = 5e-5 
    ptr = NoisyDQN().to(device)
    optimizer_ptr = torch.optim.Adam(ptr.parameters(), lr=lr)
    #target
    target_network_ptr = copy.deepcopy(ptr) #Target network copy of the original network
    target_network_ptr.load_state_dict(ptr.state_dict()) #Copies the parameters of the original model
    
    for param in target_network_ptr.parameters():
        param.requires_grad = False
    # -------------------------------#
    
    old_pos_max = 0
    global_counter = 0 #each 500 episodes launch test phase for comparison
    
    sub_goals_ig  = [[50,100], [150, 200]]
    ig = 0
    #main training loop
    while flag_get==False:

        sub_goals = []
        sub_goal_reached = False
        icm_pos = []
        
        # ----------- EXPLORATORY PHASE -----------#
        # continue till a goal is found
        while len(sub_goals) == 0:
            icm_count = 0
            
            #start exp=10 (or 1) eploratory runs
            while icm_count < icm_trials:
                print(f'---------- ICM EXPLORATION {icm_count+1} --------------')
                trajectory_icm = train_icm(env, dqn_icm, dqn_icm_target, icm_model, optimizer, buffer_icm, params, models_dict, sub_goals_dict, goal_sim, device)

                icm_pos_max_trj_i = max(trajectory_icm) #take the maximum pos reached withing an explorative trj
                icm_pos.append(icm_pos_max_trj_i) #save in a list
                icm_count += 1 #progress with the next exploratory phase
            
            c_sub_goals = list(set(icm_pos)) #avoid redundant positions
            #if explroatory runs exp=1 then the only goal available is the maximum position found. 
            if len(c_sub_goals) == 1:
                sub_goals = c_sub_goals
            #else we compute mean and std to find the interval of feasible goals
            else:
                sub_goals = filter_within_std(c_sub_goals)
            
            trj_n = None
            icm_pos_max = max(sub_goals)

        icm_pos_max = max(sub_goals)
        sub_goals = sub_goals_ig[ig]
        ig +=1
        # ----------- GOAL-CONDITIONED POLICY PHASE -----------#

        #reset weights of the last 2 linear when new position is found
        if icm_pos_max != old_pos_max:
            for module in ptr.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_parameters()  # Reinitialize parameters of NoisyLinear layers
                    module.reset_noise()
            
            # Update the target network
            target_network_ptr = copy.deepcopy(ptr)  # Deep copy of the updated network
            target_network_ptr.load_state_dict(ptr.state_dict())  # Sync parameters
            # Ensure target network parameters are not trainable
            for param in target_network_ptr.parameters():
                param.requires_grad = False
            
            optimizer_ptr = torch.optim.Adam(ptr.parameters(), lr=lr)
            
        #continue till the list of goal cadidates contains at leas one goal to try. 
        if len(sub_goals) > 0:
            while sub_goal_reached == False and len(sub_goals) > 0:
                # -------- PATH REPEATER NET ---------#
                models_dict, sub_goals_dict, trjs__plots, x_position, sub_goals, sub_goal_reached, count_model, competences, flag_get, global_counter = path_finder(ptr, target_network_ptr, optimizer_ptr, goal_sim, env, sub_goals, params, count_model, models_dict, sub_goals_dict, trjs__plots, competences, device, SEED, global_counter)
                
                # re-initialize NoisyLayers if the goal is failed, allowing for more exploration
                if sub_goal_reached == False:
                    for module in ptr.modules():
                        if isinstance(module, NoisyLinear):
                            module.reset_parameters()  # Reinitialize parameters of NoisyLinear layers
                            module.reset_noise()
                    
                    target_network_ptr.load_state_dict(ptr.state_dict())  # Sync parameters
                    # Ensure target network parameters are not trainable
                    for param in target_network_ptr.parameters():
                        param.requires_grad = False
                    
            print('\n')
            print(' --->Last X-pos: ', x_position)
            
            #update the new max position reached in the concatenated trj 
            old_pos_max = icm_pos_max
            
        
    #performance
    print('tot models', len(list(models_dict.keys())))
    print("--- %s seconds ---" % (time.time() - start_time))
    
