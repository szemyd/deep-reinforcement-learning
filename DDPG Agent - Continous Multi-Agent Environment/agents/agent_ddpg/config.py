
import torch
import torch.nn as nn
import torch.nn.functional as F

# General Training parameters #
DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")     # Training on GPU or CPU


class DDPG_AgentConfig():

    def __init__(self, 
                 BUFFER_SIZE=int(1e5),
                 UPDATE_EVERY=4,
                 BATCH_SIZE=100,
                 GAMMA=0.99,
                 LR_ACTOR=1e-4,
                 LR_CRITIC=1e-4,

                 WEIGHT_DECAY=0,
                 TAU=1e-3,
                 ACTOR_H=[32],
                 CRITIC_H=[32],
                 MU=0.,
                 THETA=0.15,
                 SIGMA=0.05,

                 OUTPUT_TYPE='probs',

                 SCORES_WINDOW=100,

                 SAVE_PATH="experiments/trained_Agents/",
                 SAVE_EXP_PATH="experiments/saved/"):

        # Replay Buffer parameters #
        self.BUFFER_SIZE = BUFFER_SIZE          # Replay Buffer size
        # Define how often the target model gets exchanged by the local model (episode num). Not used in DDPG, because we gradually mix the two models
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE            # Minibatch size

        # Learning parameters #
        self.GAMMA = GAMMA                      # Discount rate
        self.LR_ACTOR = LR_ACTOR                # Learning rate for Actor optimization
        self.LR_CRITIC = LR_CRITIC              # Learning rate for Critic optimization
        # CRITERION = F.mse_loss                # What criterion to use when comparing expected return to target return
        self.WEIGHT_DECAY = WEIGHT_DECAY        # L2 weight decay
        self.TAU = TAU                          # Target Mixin probability
        self.ACTOR_H = ACTOR_H                  # Hidden layer size of Actor Network
        self.CRITIC_H = CRITIC_H                # Hidden layer size of Critic Network

        # OUNoise parameters #
        self.MU = MU
        self.THETA = THETA
        self.SIGMA = SIGMA

        # Actor parameters #
        self.OUTPUT_TYPE = OUTPUT_TYPE

        self.SCORES_WINDOW = SCORES_WINDOW

        self.SAVE_PATH = SAVE_PATH
        self.SAVE_EXP_PATH = SAVE_EXP_PATH

    def print_constants(self):
        print("")
        print("--- General Training parameters ---")
        print("DEVICE: ", self.DEVICE)
        print("SCORES_WINDOW: ", self.SCORES_WINDOW)
        print("SAVE_PATH: ", self.SAVE_PATH)
        print("SAVE_EXP_PATH: ", self.SAVE_EXP_PATH)

        print("")
        print("--- Replay Buffer parameters ---")
        print("BUFFER_SIZE: ", self.BUFFER_SIZE)
        print("UPDATE_EVERY: ", self.UPDATE_EVERY)
        print("BATCH_SIZE: ", self.BATCH_SIZE)

        print("")
        print("--- Learning parameters ---")
        print("GAMMA: ", self.GAMMA)
        print("LR_ACTOR: ", self.LR_ACTOR)
        print("LR_CRITIC: ", self.LR_CRITIC)
        # print("CRITERION: ", CRITERION)
        print("WEIGHT_DECAY: ", self.WEIGHT_DECAY)
        print("TAU: ", self.TAU)

        print("")
        print("--- OUNoise ---")
        print("MU: ", self.MU)
        print("THETA: ", self.THETA)
        print("SIGMA: ", self.SIGMA)
