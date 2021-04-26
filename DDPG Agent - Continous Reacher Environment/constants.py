
import torch
import torch.nn as nn
import torch.nn.functional as F

# General Training parameters #
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     # Training on GPU or CPU
SCORES_WINDOW = 100
SAVE_PATH = "Trained_Agents/"
SAVE_EXP_PATH = "Experiments/"

# Replay Buffer parameters #
BUFFER_SIZE = int(1e5)          # Replay Buffer size
UPDATE_EVERY = 4                # Define how often the target model gets exchanged by the local model (episode num). Not used in DDPG, because we gradually mix the two models
BATCH_SIZE = 100                # Minibatch size

# Learning parameters #
GAMMA = 0.99                    # Discount rate
LR_ACTOR = 1e-4                 # Learning rate for Actor optimization
LR_CRITIC = 1e-4                # Learning rate for Critic optimization
CRITERION = F.mse_loss          # What criterion to use when comparing expected return to target return
WEIGHT_DECAY= 0                 # L2 weight decay
TAU = 1e-3                      # Target Mixin probability
ACTOR_H = [160, 160]              # Hidden layer size of Actor Network       
CRITIC_H =  [160, 160]            # Hidden layer size of Critic Network

# OUNoise parameters #
MU=0.
THETA=0.15
SIGMA=0.05

def print_constants():
    print("")
    print("--- General Training parameters ---")
    print("DEVICE: ", DEVICE)
    print("SCORES_WINDOW: ", SCORES_WINDOW)
    print("SAVE_PATH: ", SAVE_PATH)
    print("SAVE_EXP_PATH: ", SAVE_EXP_PATH)

    print("")
    print("--- Replay Buffer parameters ---")
    print("BUFFER_SIZE: ", BUFFER_SIZE)
    print("UPDATE_EVERY: ", UPDATE_EVERY)
    print("BATCH_SIZE: ", BATCH_SIZE)

    print("")
    print("--- Learning parameters ---")
    print("GAMMA: ", GAMMA)
    print("LR_ACTOR: ", LR_ACTOR)
    print("LR_CRITIC: ", LR_CRITIC)
    print("CRITERION: ", CRITERION)
    print("WEIGHT_DECAY: ", WEIGHT_DECAY)
    print("TAU: ", TAU)

    print("")
    print("--- OUNoise ---")
    print("MU: ", MU)
    print("THETA: ", THETA)
    print("SIGMA: ", SIGMA)
