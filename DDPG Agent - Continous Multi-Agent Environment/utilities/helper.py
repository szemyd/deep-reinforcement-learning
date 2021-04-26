import os.path
import torch
# from constants import *             # Capital lettered variables are constants from the constants.py file


# def get_constant_string(config):
#     for_title = "Network:: A: {} | C: {}\nLearning:: LR_A: {} | LR_C: {} | TAU: {} \nNoise:: MU: {} | THETA: {} | SIGMA: {}\nBuffer:: Size: {} | Batch size: {}".format(config.ACTOR_H, config.CRITIC_H, config.LR_ACTOR, config.LR_CRITIC, config.TAU, config.MU, config.THETA, config.SIGMA, config.BUFFER_SIZE, config.BATCH_SIZE)
#     for_filename = "A_{} - C_{}".format(' '.join([str(elem) for elem in config.ACTOR_H]), ' '.join([str(elem) for elem in config.CRITIC_H]))
#     return for_title, for_filename




def fileAtLocation(filename, path):
    return os.path.exists(path + filename)

def load_previous(agent, filename, path):
    filename_actor = filename +'_actor.pth'
    filename_critic = filename + '_critic.pth'

    loaded_agent = agent
    if fileAtLocation(filename_actor, path):
        print("Found previous trained Agent with same neural nets, going to load them!")
        loaded_agent.actor_local.load_state_dict(torch.load(path + filename_actor))
        loaded_agent.critic_local.load_state_dict(torch.load(path + filename_critic))
    else:
        print("Didn't find any saved agents")
    return loaded_agent



def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def unique(a):
    return list(set(a))

def mean(lst):
    return sum(lst) / len(lst)