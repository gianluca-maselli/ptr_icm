import torch
from collections import deque
from IntrinsicCuriosityModule.utils import get_action
from IntrinsicCuriosityModule.icm import compute_extrinsic_reward, minibatch_train
from IntrinsicCuriosityModule.utils import *
from IntrinsicCuriosityModule.test_icm import test_icm


def train_icm(env, dqn_icm, dqn_icm_target, icm_model, optimizer, icm_buffer, params, device):
    
    max_episodes = 100000 #use a value higher than the avg of experiments with ICM+PTR
    n_frames = 4
    queue = deque(maxlen=n_frames)
    #get first state
    init_frame = env.reset()
    frame_queue = initialize_queue(queue, 4, init_frame)
    input_frames = stack_frames(frame_queue)
    state = input_frames.unsqueeze(0).permute(0,3,1,2)

    done = False
    target_update_freq = 1e4    
    episode = 0
    step = 0

    #for epoch in range(1, max_epochs):
    while episode < max_episodes+1: 
        step +=1
        #Runs the DQN forward to get action-value predictions
        q_val_pred = dqn_icm(state.to(device))
        #get action
        action = get_action(q_val_pred)
        #env step
        next_frame, e_reward_, done, info = skip_frames(action,env, skip_frame=params['action_repeats_icm'])
        
        #give additional extrinsic bonus to the end of the level
        if info['x_pos'] >= 3000:
            ex_reward = 1
            done=True
        else:
            ex_reward = 0

        frame_queue.append(frame_preprocessing(next_frame))
        next_state = stack_frames(frame_queue).unsqueeze(0).permute(0,3,1,2)

        curiosity_signal = compute_extrinsic_reward(icm_model, state, next_state, action, q_val_pred.shape[1], params['eta'], device)
        
        #add extrinsic to the intrinsic to add the final reward.
        curiosity_signal = curiosity_signal + ex_reward
        icm_buffer.add((state, action, torch.from_numpy(curiosity_signal), next_state, done))

        if done:
            episode +=1
            #run test phase each 500 episodes for comparison
            if episode % 500  == 0:
                _, _, last_t_pos = test_icm(env, dqn_icm, params, device)
                print(f'Episode: {episode}, Pos:{last_t_pos}')
                dqn_icm.train()
            
            #re-initialize env
            init_frame = env.reset()
            frame_queue = initialize_queue(queue, 4 , init_frame)
            input_frames = stack_frames(frame_queue)
            next_state = input_frames.unsqueeze(0).permute(0,3,1,2)

            done = False


        state = next_state
        if icm_buffer.len() < params['batch_size']:
            continue

        # ------- Double-DQN TRAINING -------- #
        inverse_loss, forward_loss, dqn_loss, loss = minibatch_train(icm_buffer, dqn_icm, dqn_icm_target, icm_model, optimizer, params['batch_size'], q_val_pred.shape[1], params, device)
        
        if step % target_update_freq == 0:
            dqn_icm_target.load_state_dict(dqn_icm.state_dict())
