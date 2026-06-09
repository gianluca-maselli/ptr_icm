import torch
from IntrinsicCuriosityModule.utils import *
from collections import deque
from IntrinsicCuriosityModule.utils import get_action

def test_icm(env, model, params, device):

    model.eval()

    queue = deque(maxlen=4)
    in_state = env.reset()
    frame_queue = initialize_queue(queue, 4 , in_state)
    input_frames = stack_frames(frame_queue)
    current_state = input_frames.unsqueeze(0).permute(0,3,1,2)

    ep_len = 0
    
    render = []
    list_pos = []
    icm_trj = []
    while True:
        ep_len +=1
        with torch.no_grad():
            q_val_pred = model(current_state.to(device))
        
        action = get_action(q_val_pred)
        next_frame, e_reward_, done, info  = skip_frames(action,env, skip_frame=params['action_repeats_icm'])
        icm_trj.append(next_frame.copy())
        list_pos.append(info['x_pos'])
        
        render.append(next_frame.copy())
        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue).unsqueeze(0).permute(0,3,1,2)

        if done:
            #print('tot steps', ep_len)
            #print('max pos', max(list_pos))
            break

        current_state = next_state
    return icm_trj, list_pos, info['x_pos']
