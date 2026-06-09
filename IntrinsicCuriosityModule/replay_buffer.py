import random
import torch
import numpy as np

class ReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size, device):
        # state, action, reward, next_state, done
        self.state = torch.empty(state_size, dtype=torch.float)
        self.action = torch.empty(buffer_size, action_size, dtype=torch.long)
        self.reward = torch.empty(buffer_size, dtype=torch.float)
        self.next_state = torch.empty(state_size, dtype=torch.float)
        self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
        self.device = device
        self.state_size = state_size
        self.action_size = action_size

    def add(self, transition):
        state, action, reward, next_state, done = transition
        # store transition in the buffer
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size
        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)
        batch = (
                self.state[sample_idxs].to(self.device),
                self.action[sample_idxs].to(self.device),
                self.reward[sample_idxs].to(self.device),
                self.next_state[sample_idxs].to(self.device),
                self.done[sample_idxs].to(self.device),
                )
        return batch

    def len(self,):
        return self.real_size

    def refresh(self,):
        self.state = torch.empty(self.state_size, dtype=torch.float)
        self.action = torch.empty(self.size, self.action_size, dtype=torch.long)
        self.reward = torch.empty(self.size, dtype=torch.float)
        self.next_state = torch.empty(self.state_size, dtype=torch.float)
        self.done = torch.empty(self.size, dtype=torch.int)

        self.count = 0
        self.real_size = 0


