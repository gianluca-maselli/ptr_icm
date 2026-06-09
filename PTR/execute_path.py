import cv2
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os
import imageio

import matplotlib.pyplot as plt
import numpy as np

from PTR.experience_replay import *
from PTR.utils import *
from PTR.recover_path import repeat_path

os.environ['OMP_NUM_THREADS'] = '1'

def save_current_model(count_model, prt_model, sub_goal, models_dict, sub_goals_dict, trjs__plots, test_pos, c_array, competences, render_test, seed):
    print(' ------- Test finished ------- ')
    print('\n')
    print(' ------- SAVE MODEL ------- ')
    fps = 8
    count_model +=1
    path_save = './model'+str(count_model)+'_seed'+str(seed)+'.pt'
    torch.save(prt_model, path_save)
    print(f'added model {str(count_model)}')
    #save model in the dict
    name = 'model'+str(count_model)
    models_dict[name] = path_save
    #save subgoal associated to the model
    sub_goals_dict[name] = sub_goal
    trjs__plots['Goal '+str(count_model)] = test_pos
    competences['Goal '+str(count_model)] = c_array
    #save gif of the current sub path
    file_gif_sub = './gifs_goals/gif_Test_phase_r'+str(count_model)+'_seed'+str(seed)+'.gif'
    if os.path.isfile(file_gif_sub):
        os.remove(file_gif_sub)
    imageio.mimsave(file_gif_sub, [np.array(img_j) for img_j in render_test], fps = fps)
    
    print('models_dict: ', models_dict)
    print('sub_goals_dict: ', sub_goals_dict)
    print('trjs__plots: ', trjs__plots)
    print('competences: ', competences)

    return count_model, models_dict, sub_goals_dict, trjs__plots, competences

# ----- train ------
def path_finder(model, target_model, optimizer, goal_sim, env, sub_goals, params, count_model, models_dict, sub_goals_dict, trjs__plots, competences, device, seed, global_counter):

    model.train()
    target_model.eval()
    
    max_episodes = int(1e4)
    
    # --- Noisy Net hyperparams --- #
    update_interval = 1e4
    burnin = 32  
    train_interval = 1
    max_size_buffer_ptr = int(1e5)
    batch_size = 32
    gamma = 0.99

    sub_goal_reached = False
    test_x_pos = []
    
    # ---- Competence(C) ----- # 
    comp_w_size = 30
    comp_tresh = 0.9 #competence treshold
    comp_window = deque(maxlen=comp_w_size) #competence window over the last 30 episodes
    C = 0 #competence measure
    c_array = [] #keep track of the competences for a sub-goal
    tot_reward = 0
    
    reached_1st_time = False
    flag_get = False
    

    # ---- PER ----- #
    replay = PER(size=max_size_buffer_ptr, batch_size=batch_size)
    reached_1st_time = False
    beta_start = 0.4
    beta_ep_ = 1e5
    beta_annealing = lambda episode_i: min(1.0, beta_start + episode_i * (1.0 - beta_start) / beta_ep_)
    steps_beta = 0
    
    queue = deque(maxlen=4)
    steps = 0
    x_pos = 0
    
    #sub-goal selection
    sub_goal = max(sub_goals)
    sub_goal_i = sub_goals.index(sub_goal)
    print('\n')
    print(f'----> Sub-Goal: {sub_goal}')
    
    #similarity class
    for episode in range(1, max_episodes+1):

        ep_len = 0
        reached = False
        tot_reward = 0
        done = False
        done_steps = False
        test_phase = False
        render = []
        #increase for testing each 500 episodes
        global_counter +=1 

        #check which last sub-goal has to be reach before starting training 
        if len(models_dict.keys()) == 0:
            in_state = env.reset()
        else:
            _, in_state, x_p, _, _, _, _ = repeat_path(env, None, goal_sim, models_dict, None, sub_goals_dict, params, test_phase, device)
            x_pos = x_p
                    

        frame_queue = initialize_queue(queue, 4 , in_state)
        input_frames = stack_frames(frame_queue)
        current_state = input_frames.unsqueeze(0).permute(0,3,1,2)

        #set the current goal  
        goal_sim.set_goal(sub_goal)
        
        #episode start
        while True:
            steps +=1
            ep_len +=1
            
            with torch.no_grad():
                q_vals_ = model(current_state.to(device))

            #select action with noise
            action = act_noisy(q_vals_)
            #step
            next_frame, reward, done, info, reached, flag_reached  = skip_frames(action, env, goal_sim, skip_frame=params['action_repeats_ptr'])
            
            frame_queue.append(frame_preprocessing(next_frame))
            next_state = stack_frames(frame_queue).unsqueeze(0).permute(0,3,1,2)
            render.append(next_frame.copy())
            tot_reward +=reward
            
            #truncate after max steps
            if (ep_len >= params['max_episode_len']  and reached == False):
                done_steps = True

            if reached:
                #we want to get at least 1 positive reward before starting beta anneling in PER
                if steps >= burnin:
                    reached_1st_time = True

            replay.push(current_state, action, reward, next_state, done)
            current_state = next_state

            #learn
            # ------ TARGET NET SYN ------- #
            if steps % update_interval == 0:
                target_model.load_state_dict(model.state_dict())
           
            # -------------- UPDATE ---------------# 
            if steps >=  burnin and steps % train_interval == 0:
                if reached_1st_time:
                    steps_beta +=1
                    beta_used = beta_annealing(steps_beta)
                else:
                    beta_used = beta_start
                state_batch, action_batch, reward_batch, state2_batch, done_batch,  weights, indices  = replay.sample(beta_used)
                
                current_Q = model(state_batch.to(device))[np.arange(0, batch_size), action_batch]
                with torch.no_grad():
                    next_state_Q = model(state2_batch.to(device))
                    best_action = torch.argmax(next_state_Q, dim=1)
                    next_Q = target_model(state2_batch.to(device))[np.arange(0, batch_size), best_action]
                expected_Q = (reward_batch.to(device) + (1 - done_batch.to(device)) * gamma  * next_Q).float()
                
                if weights is None:
                    weights = torch.ones_like(current_Q)
                    
                td_error = (expected_Q.detach()- current_Q)
                losses  = td_error.pow(2) * weights.to(device)
                prios =  losses + 1e-5 
                loss  = losses.mean()
                 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #update priorities in PER 
                replay.update_priorities(indices, prios.detach().cpu().numpy())
            
            model.reset_noise()
            target_model.reset_noise()

            #DONE: stop the current episode
            if done or done_steps:
                comp_window.append(int(reached))
                if len(comp_window) >=comp_w_size:
                    #update current competence
                    C = sum(comp_window)/len(comp_window)
                c_array.append(C)
                
                print(f"Episode:{episode}, Competence(C): {C}")
                
                #perform test after 500 episodes for comparision 
                if global_counter  % 500  == 0:
                    test_phase = True
                    _, _, x_pos_test, _, _, _, _ = repeat_path(env, model, goal_sim, models_dict, sub_goal, sub_goals_dict, params, test_phase, device)
                    test_x_pos.append(x_pos_test)
                    
                break

        #check condition for run testing phase
        if C >= comp_tresh:
            test_phase = True
            sub_goal_reached, next_frame, x_p, test_len, test_pos, render_test, flag_get = repeat_path(env, model, goal_sim, models_dict, sub_goal, sub_goals_dict, params, test_phase, device)
            
            if sub_goal_reached or flag_get:
                count_model, models_dict, sub_goals_dict, trjs__plots, competences= save_current_model(count_model, model, sub_goal, models_dict, sub_goals_dict, trjs__plots, test_pos, c_array, competences, render_test, seed)
                x_pos = x_p
                break
        
        model.train()
        
        #if goal too difficult, then discard 
        if (episode >= max_episodes):
            print('\n')
            print('----- path not found -----')
            print('----> goal discarded: ', sub_goal_i)
            sub_goals.pop(sub_goal_i)
            print('--------------------------------')
            break
    
    print('Test Pos:', test_x_pos)
    return models_dict, sub_goals_dict, trjs__plots, x_pos, sub_goals, sub_goal_reached, count_model, competences, flag_get, global_counter
