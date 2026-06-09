import torch
from torchvision import transforms

import numpy as np
import cv2
import umap
import pandas as pd
import shutil
import os

# ---- GOAL SIMILARITY ---- #

#used to check at each step if the sub-goal is reached and get the reward = 1
class GoalSimilarity:
    def __init__(self):
        #set current goal 
        self.goal = None
    
    def __reward_target__(self, x_current):
        reached = False
        goal_int = 15
        if x_current >= self.goal and x_current <= self.goal + goal_int:
            reward = 1
            reached = True
        else:
            reward = 0
        return reward, reached

    def set_goal(self, goal):
        self.goal = goal 
