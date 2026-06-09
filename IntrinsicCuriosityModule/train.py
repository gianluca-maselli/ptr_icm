import torch
from collections import deque
from IntrinsicCuriosityModule.utils import get_action #, get_temperature_icm, boltzman_exploration_icm
from IntrinsicCuriosityModule.icm import compute_extrinsic_reward, minibatch_train
from IntrinsicCuriosityModule.utils import *
from PTR.recover_path import repeat_path


def train_icm(env, dqn_icm, dqn_icm_target, icm_model, optimizer, icm_buffer, params, models_dict, sub_goals_dict, goal_sim, device):
    
    max_epochs = int(10e6) #use only for experiments with ICM only
    n_frames = 4
    queue = deque(maxlen=n_frames)
    test_phase = False

    # by using PTR continue the exploration from the last sub-goal found
    if len(models_dict.keys()) == 0:
        init_frame = env.reset()
        #recover path till the last executed
    else:
        _, init_frame, _, _, _, _, _ = repeat_path(env, None, goal_sim, models_dict, None, sub_goals_dict, params, test_phase, device)
        
    #get first state
    frame_queue = initialize_queue(queue, 4 , init_frame)
    input_frames = stack_frames(frame_queue)
    state = input_frames.unsqueeze(0).permute(0,3,1,2)

    episode_length = 0
    done = False
    target_update_freq = 1e3 
    icm_trj = []

    for epoch in range(max_epochs):
        episode_length +=1
        #Runs the DQN forward to get action-value predictions
        q_val_pred = dqn_icm(state.to(device))
        #action selection
        action = get_action(q_val_pred)
        #env step
        next_frame, e_reward_, done, info = skip_frames(action,env, skip_frame=params['action_repeats_icm'])
        
        #keep track of the positions
        icm_trj.append(info['x_pos'])

        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue).unsqueeze(0).permute(0,3,1,2)

        curiosity_signal = compute_extrinsic_reward(icm_model, state, next_state, action, q_val_pred.shape[1], params['eta'], device)

        icm_buffer.add((state, action, torch.from_numpy(curiosity_signal), next_state, done))

        if done: # or done_steps:
            print(f'Len Ep: {episode_length}, Max Pos: {max(icm_trj)}')
            return icm_trj #return the sequence of positions in the trajecotry
    
        state = next_state
        if icm_buffer.len() < params['batch_size']:
            continue

        # ------- Double-DQN TRAINING -------- #
        inverse_loss, forward_loss, dqn_loss, loss = minibatch_train(icm_buffer, dqn_icm, dqn_icm_target, icm_model, optimizer, params['batch_size'], q_val_pred.shape[1], params, device)
        
        if epoch % target_update_freq == 0:
            dqn_icm_target.load_state_dict(dqn_icm.state_dict()) 
