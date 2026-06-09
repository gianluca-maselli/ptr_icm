import numpy as np 
import torch
import cv2
from collections import deque
from torch.nn import init
import torch.nn.functional as F

# ------- ICM MODELS AND UTILS -------- #
def compute_extrinsic_reward(icm_model, state, next_state, action, output_size, eta, device):
    action = np.array([action])
    action = torch.LongTensor(action).to(device) #.unsqueeze(0
    #get one_hot_encoding
    action_onehot = torch.FloatTensor(len(action), output_size).to(device) 
    action_onehot.zero_()
    action_onehot.scatter_(1, action.view(len(action), -1), 1)
    real_next_state_feature, pred_next_state_feature, pred_action = icm_model([state.to(device), next_state.to(device), action_onehot.to(device)])
    intrinsic_reward = eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1)
    if device == 'cuda':
        intrinsic_reward = intrinsic_reward.detach().cpu().numpy()
    else:
        intrinsic_reward = intrinsic_reward.data.numpy()

    return intrinsic_reward

# -------- MINIBATCH TRAINING ------- #
def minibatch_train(icm_buffer, dqn, dqn_target, icm_model, optimizer, batch_size, output_size, params, device):
    batch = icm_buffer.sample(batch_size)
    state_batch, action_batch, intrinsic_rw_batch, next_state_batch, done_batch = batch
    cross_entropy = torch.nn.CrossEntropyLoss()
    forward_mse = torch.nn.MSELoss()

    action_onehot_batch = torch.FloatTensor(action_batch.shape[0], output_size).to(device)
    action_onehot_batch.zero_()
    action_onehot_batch.scatter_(1, action_batch.view(-1, 1), 1)
    #print('action_onehot_batch shape', action_onehot_batch.shape)
    #print('action_onehot_batch', action_onehot_batch)
    real_next_state_feature, pred_next_state_feature, pred_action = icm_model([state_batch.to(device), next_state_batch.to(device), action_onehot_batch.to(device)])
    # ------- INVERSE LOSS (S_t, S_t+1 = a_t+1)------ #
    inverse_loss = cross_entropy(pred_action, action_batch.reshape([-1]))
    # ------- FORWARD LOSS (S_t, a_t = S_t+1)------ #
    forward_loss = forward_mse(pred_next_state_feature, real_next_state_feature.detach())
    # ----- DOUBLE DQN UPDATE ----- #
    #compute q_values for the batch of observations
    q_values = dqn(state_batch.to(device))
    q_sel_action = q_values.gather(dim=1, index=action_batch) #take the q_value of the corresponding action in the batch
    #get the q_value for the best action for next states
    q_values_next = dqn(next_state_batch.to(device)).detach()
    _, a_prime = q_values_next.max(1)
    #get Q values from target net for next state and chosen action
    q_target_values_next = dqn_target(next_state_batch.to(device)).detach()
    q_target_s_a_prime = q_target_values_next.gather(1, a_prime.unsqueeze(1))
    # if current state is end of episode, then there is no next Q value
    q_target_s_a_prime = (1 - done_batch).unsqueeze(1) * q_target_s_a_prime 
    expected_q_value = (intrinsic_rw_batch.unsqueeze(1) + params['gamma'] * q_target_s_a_prime)
    dqn_loss = (q_sel_action - expected_q_value).pow(2).mean()
    loss = ((1- params['beta']) * inverse_loss) + (params['beta'] * forward_loss) + (params['lambda'] * dqn_loss) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return inverse_loss, forward_loss, dqn_loss, loss

# --------- ICM MODEL (ENCODER, INVERSE MODEL, FORWARD MODEL) ---------- #
class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_planes, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.elu = torch.nn.ELU()

    def forward(self, x):
        out = self.conv(x)
        out = self.elu(out)
        return out

class ICM(torch.nn.Module):
    def __init__(self, input_shape, layers, kernel_sizes, strides, fc_dim, out_dim, padding, device):
        super(ICM, self).__init__()
        self.device = device
        # ----- ENCODER MODEL ------ #
        self.encoder = torch.nn.Sequential(
                ConvBlock(in_planes=input_shape, out_channels=layers[0], kernel_size=kernel_sizes[0], stride=strides[0], padding=padding[0]),
                ConvBlock(in_planes=layers[0], out_channels=layers[1], kernel_size=kernel_sizes[1], stride=strides[1], padding=padding[0]),
                ConvBlock(in_planes=layers[1], out_channels=layers[2], kernel_size=kernel_sizes[2], stride=strides[2], padding=padding[0]),
                ConvBlock(in_planes=layers[2], out_channels=layers[3], kernel_size=kernel_sizes[3], stride=strides[3], padding=padding[0]),
                torch.nn.Flatten()
                )
        # ------- INVERSE MODEL -------- #
        self.inverse_model = torch.nn.Sequential(
                torch.nn.Linear(288*2, fc_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(fc_dim, out_dim)
                )
        # ------- FORWARD MODEL --------#
        self.residual = [torch.nn.Sequential(
            torch.nn.Linear(out_dim+288, fc_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(fc_dim, 288)).to(self.device)]*8
        
        self.forward_model1 = torch.nn.Sequential(
                torch.nn.Linear(out_dim+288, 288),
                torch.nn.LeakyReLU()
                )
        self.forward_model2 = torch.nn.Sequential(
                torch.nn.Linear(out_dim+288, 288),
                )
        
        '''
        self.forward_model = torch.nn.Sequential(
        torch.nn.Linear(out_dim+288, fc_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(fc_dim, 288),
        )
        '''

        for p in self.modules():
            if isinstance(p, torch.nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, torch.nn.Linear):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs
        #encode state and next state
        encoded_state = self.encoder(state)
        encoded_next_state = self.encoder(next_state)
        #get predicted action
        pred_action = torch.cat((encoded_state, encoded_next_state), 1)
        pred_action = self.inverse_model(pred_action)
        #next state prediction
        pred_next_state_feature_orig = torch.cat((encoded_state, action), 1)
        #pred_next_state_feature = self.forward_model(pred_next_state_feature_orig)
        pred_next_state_feature_orig = self.forward_model1(pred_next_state_feature_orig)

        # residual forward model inspired by: https://github.com/jcwleo/curiosity-driven-exploration-pytorch/blob/master/model.py
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig
    
        pred_next_state_feature = self.forward_model2(torch.cat((pred_next_state_feature_orig, action), 1))
    
        real_next_state_feature = encoded_next_state
    
        return real_next_state_feature, pred_next_state_feature, pred_action


