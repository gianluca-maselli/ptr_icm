from PTR.test import test_ptr
import torch 
import cv2

def repeat_path(env, ptr_model, sim_class, models_dict, new_goal, sub_goals_dict, params, test_phase, device):

    #initialize the env
    init_frame = env.reset()
    test_completed = False     
    if len(models_dict.keys()) > 0:
        for model_name in models_dict.keys():
            #print('model loaded: ', model_name)
            path_model = models_dict[model_name]
            sub_goal_i = sub_goals_dict[model_name]
            ptr_model_i = torch.load(path_model)
            dead_check = False
            #set the current goal
            sim_class.set_goal(sub_goal_i)
            
            _, init_frame, last_x_pos, test_len, test_pos, render, flag_get = test_ptr(env, ptr_model_i, init_frame, sim_class, params, dead_check,  device)
            #print('-------------------------------')
        
        if test_phase:
            #print('\n')
            #print('----- TEST PHASE AFTER PATH RECOVER -----')
            sim_class.set_goal(new_goal)
            dead_check = True
            test_completed, _ , last_x_pos, test_len, test_pos, render, flag_get = test_ptr(env, ptr_model, init_frame, sim_class, params, dead_check, device)
        
    else:
        #print('----- First Time of Test Phase -----')
        sim_class.set_goal(new_goal)
        dead_check = True
        test_completed, init_frame, last_x_pos, test_len, test_pos, render, flag_get = test_ptr(env, ptr_model, init_frame, sim_class, params, dead_check, device)

    return test_completed, init_frame, last_x_pos, test_len, test_pos, render, flag_get

