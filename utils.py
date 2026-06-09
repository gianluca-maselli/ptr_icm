import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import itertools

def skip_frames(action,env, skip_frame=6):
    skipped_frame = deque(maxlen=2)
    skipped_frame.clear()
    total_reward = 0.0
    done = None
    for _ in range(skip_frame):
        n_state, reward, done, info = env.step(action)
        last_x_pos = info['x_pos']
        skipped_frame.append(n_state)
        total_reward += reward
        if done:
            break
    max_frame = np.max(np.stack(skipped_frame), axis=0)
    #max_frame = n_state
    return max_frame, total_reward, done, info, last_x_pos

class MarioDataset(Dataset):
    def __init__(self, image_array ,transform=None):
        self.images = image_array  
        self.transform = transform  
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = self.images[index]
        
        if self.transform is not None:
            aug = self.transform(image=img)
            image = aug['image']
        
        return image
    
transforms_m = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


def plot_goals(goal_dict, x_ax, y_ax, mode, exp_t, file_name, folder):
    # Create a color map for distinct colors
    cmap = plt.get_cmap('tab20')

    # Set up the plot
    plt.figure(figsize=(10, 6))
    steps_count = 0
    joint_trjs = []
    colors = []
    labels = []  # To store labels for individual points
    ptr_trajectory_label_added = False  # To ensure 'PTR Trajectory' label is added only once

    # Plot each goal with a unique color
    for idx, (goal, values) in enumerate(goal_dict.items()):
        color = cmap(idx % 10)  # Get color for the current index
        if mode == 'ptr':
            colors.append(color)
            len_val = len(values)
            steps_count += len_val
            joint_trjs.append(list(values))

            # Add a label for this point in the format 'goal: pos'
            labels.append((goal, values[-1], color))  # Store label and color for later
        else:
            plt.plot(values, label=f"Goal {idx+1}: {goal}", color=color)

    if mode == 'ptr':
        # Prepare steps and joint_trjs for plotting
        steps = [step for step in range(steps_count)]
        joint_trjs = list(itertools.chain(*joint_trjs))
        joint_trjs = np.array(joint_trjs)

        # Plot the full trajectory line
        plt.plot(steps, joint_trjs, label='PTR Trajectory', color='black')  # Color the trajectory line

        # Plot dots at the end of each array and add individual goal labels
        current_step = 0
        for i, (goal, values) in enumerate(goal_dict.items()):
            end_step = current_step + len(values) - 1  # Index of the last element
            plt.plot(steps[end_step], joint_trjs[end_step], 'o', color=colors[i])  # Plot dot
            current_step += len(values)

        # Add each goal:pos label to the legend
        for goal, pos, color in labels:
            plt.plot([], [], 'o', color=color, label=f"{goal}: {pos}")

    plt.xlim(0)

    # Update the legend with PTR Trajectory and individual points
    plt.grid(True)  # Enable the grid
    plt.xlabel(x_ax)
    plt.ylabel(y_ax)
    #plt.title(title)
    plt.legend(loc='lower right')

    # Show the plot
    if exp_t == 'w_r':
        plt.savefig(folder+'/plot_w_r_'+str(file_name)+'.png')
    else:
        plt.savefig(folder+'/plot_'+str(file_name)+'.png')
    #plt.show()

def running_mean(x,N=100):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


def plot_icm_reward(list_rewards, episode):
    cmap = plt.get_cmap('tab20')
    list_rewards = running_mean(list_rewards,N=3)
    # Set up the plot
    plt.figure(figsize=(10, 6))
    color = cmap(5 % 10)
    plt.plot(list_rewards, label=f"avg reward", color=color)
    plt.xlim(0)
    plt.grid(True)  # Enable the grid
    plt.xlabel('Steps')
    plt.ylabel('average intrinsic reward')
    plt.title('AVG ICM Intrinsic Reward (no weights init)')
    #plt.show()
    plt.savefig('./exp_plots/plot_icm_reward_'+str(episode)+'.png')
