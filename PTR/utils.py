import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import imageio

# --- FRAMES PROCESSING ---- #
def frame_preprocessing(frame):
    img = np.reshape(frame, [256, 240, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

def initialize_queue(queue, n_frames, init_frame):
    queue.clear()
    for i in range(n_frames):
        queue.append(frame_preprocessing(init_frame))
    return queue

def skip_frames(action,env, sim_class, skip_frame=4):
    
    skipped_frame = deque(maxlen=2)
    skipped_frame.clear()
    total_reward = 0.0
    done = False

    for _ in range(skip_frame):
        n_state, _ , dead_, info = env.step(action) #here done is only for deaths
        
        x_current = info['x_pos']
        flag_get = info['flag_get']
        #check if the agent is at the end of the level
        if x_current > 3000:
            flag_get = True
        
        reward, reached  = sim_class.__reward_target__(x_current)
        skipped_frame.append(n_state)
        total_reward += reward
        
        #dead is basically the 'done' condtion in gym, so at the end of the level we got 'done' as well 
        if dead_ and flag_get == False:
            total_reward = 0 
            reached = False            
            done = True
            break

        if flag_get:
            reached = True

        if reached:
            done = True
            break    
    max_frame = np.max(np.stack(skipped_frame), axis=0)

    return max_frame, total_reward, done, info, reached, flag_get

def stack_frames(stacked_frames):
    #concatenate the frames
    frames_stack = np.concatenate(stacked_frames, axis=-1)
    frames_stack = frames_stack.astype(np.float32) / 255.0
    return torch.tensor(frames_stack, dtype=torch.float32)

# ---- ACTION SELECTION -------
def act_noisy(q_vals):
    action  = np.argmax(q_vals.detach().cpu().numpy())
    return action

