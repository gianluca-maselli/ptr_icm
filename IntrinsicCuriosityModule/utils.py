import numpy as np 
import torch
import cv2
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------ FRAMES AND ENV PREPROCESSING ------- #
def frame_preprocessing(frame):
    img = np.reshape(frame, [256, 240, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    x_t = cv2.resize(img, (42, 42), interpolation=cv2.INTER_AREA)
    x_t = np.reshape(x_t, [42, 42, 1])
    return x_t.astype(np.uint8)

def initialize_queue(queue, n_frames, init_frame):
    queue.clear()
    #init_frame = Noop(env, actions_name, noop_max=30)
    for i in range(n_frames):
        queue.append(frame_preprocessing(init_frame))
    return queue

def stack_frames(stacked_frames):
    #concatenate the frames
    frames_stack = np.concatenate(stacked_frames, axis=-1)
    frames_stack = frames_stack.astype(np.float32) / 255.0
    return torch.tensor(frames_stack, dtype=torch.float32)


def skip_frames(action,env, skip_frame=6):
    skipped_frame = deque(maxlen=2)
    skipped_frame.clear()
    e_reward_tot = 0.0 #extrinsic reward
    done = None
    for _ in range(skip_frame):
        n_state, e_reward, done, info = env.step(action)
        if info['flag_get']:
            done=True
        skipped_frame.append(n_state)
        e_reward_tot += e_reward
        if done:
            break
    max_frame = np.max(np.stack(skipped_frame), axis=0)

    return max_frame, e_reward_tot, done, info

def random_choice_prob_index(p, axis=1):
    r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis)
    return (p.cumsum(axis=axis) > r).argmax(axis=axis)


def get_action(q_values):
    action_prob = F.softmax(q_values, dim=-1).detach().cpu().numpy()
    action = random_choice_prob_index(action_prob)[0]
    #print('action', action)
    return action

def running_mean(x,N=100):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


def plot_icm_reward(list_rewards, episode):
    cmap = plt.get_cmap('tab10')
    list_rewards = running_mean(list_rewards,N=100)
    # Set up the plot
    plt.figure(figsize=(10, 6))
    color = cmap(5 % 10)
    plt.plot(list_rewards, label=f"avg reward", color=color)
    plt.xlim(0)
    plt.grid(True)  # Enable the grid
    plt.xlabel('Episode')
    plt.ylabel('average intrinsic reward')
    plt.title('AVG ICM Intrinsic Reward (no weights init)')
    #plt.show()
    plt.savefig('./exp_plots/plot_icm_reward_'+str(episode)+'.png')
