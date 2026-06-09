import torch 
import numpy as np
from PTR.utils import *
import os 
import imageio

#-------- test ----------

#check if mario after the testing trajectory it is stuck in a death loop
def death_check_fun(env):
    reached = True
    done = False
    for _ in range(60):
        _, _, done, info = env.step(0)
        if done==True:
            reached = False
            done = True
            break
    return reached, done

def test_ptr(env, model, init_frame, sim_class, params, dead_check, device):
    
    model.eval()

    queue = deque(maxlen=4)
    
    frame_queue = initialize_queue(queue, 4 , init_frame)
    input_frames = stack_frames(frame_queue)
    current_state = input_frames.unsqueeze(0).permute(0,3,1,2)
    reached = False 
    ep_len = 0
    last_x_pos = 0
    
    render = []
    test_pos = []
    while True:
        ep_len +=1
        with torch.no_grad():
            q_val = model(current_state.to(device))
        
        action = np.argmax(q_val.detach().cpu().numpy())
        next_frame, reward, done, info, reached, flag_get = skip_frames(action, env, sim_class, skip_frame=params['action_repeats_ptr'])
        render.append(next_frame.copy())
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue).unsqueeze(0).permute(0,3,1,2)
        last_x_pos = info['x_pos']
        test_pos.append(last_x_pos)
        
        if ep_len >= params['max_episode_len']:
            break

        if reached:
            if flag_get and dead_check:
                reached = True
                done = False
            elif not flag_get and dead_check:
                reached, done = death_check_fun(env)    
            break
        
        if reached or done:
            break

        current_state = next_state

    return reached, next_frame, last_x_pos, ep_len, test_pos, render, flag_get
