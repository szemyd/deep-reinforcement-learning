import os.path
import torch
from constants import *             # Capital lettered variables are constants from the constants.py file


def get_constant_string():
    for_title = "Network:: A: {} | C: {}\nLearning:: LR_A: {} | LR_C: {} | TAU: {} \nNoise:: MU: {} | THETA: {} | SIGMA: {}\nBuffer:: Size: {} | Batch size: {}".format(ACTOR_H, CRITIC_H, LR_ACTOR, LR_CRITIC, TAU, MU, THETA, SIGMA, BUFFER_SIZE, BATCH_SIZE)
    for_filename = "A_{} - C_{}".format(' '.join([str(elem) for elem in ACTOR_H]), ' '.join([str(elem) for elem in CRITIC_H]))
    return for_title, for_filename


_, filename = get_constant_string() ##"checkpoint_actor.pth"
filename_actor = filename +'_actor.pth'
filename_critic = filename + '_critic.pth'
path = SAVE_PATH

def fileAtLocation(filename, path):
    return os.path.exists(path + filename)

def load_previous(new_agent):
    loaded_agent = new_agent
    if fileAtLocation(filename_actor, path):
        print("Found previous trained Agent with same neural nets, going to load them!")
        loaded_agent.actor_local.load_state_dict(torch.load(SAVE_PATH + filename_actor))
        loaded_agent.critic_local.load_state_dict(torch.load(SAVE_PATH + filename_critic))
    else:
        print("Didn't find any saved agents")
    return loaded_agent


