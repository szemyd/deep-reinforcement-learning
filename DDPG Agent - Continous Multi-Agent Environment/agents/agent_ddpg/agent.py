## Deep Deterministic Policy Gradients ##
from agents.abstract_agent import Agent
from .model import Actor, Critic    # These are our models
# from model_provided import Actor, Critic    # These are our models
import numpy as np
import random                       # Used for random seed
# This is used for the mixing of target and local model parameters
import copy

# from .constants import *             # Capital lettered variables are constants from the constants.py file
# Our replaybuffer, where we store the experiences
from .memory import ReplayBuffer
from .config import DDPG_AgentConfig

import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
import os
sys.path.append(os.path.abspath('..'))


DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")     # Training on GPU or CPU


class DDPG_Agent(Agent):
    def __init__(self, state_size, action_size, seed=1, config=DDPG_AgentConfig()):
        super(DDPG_Agent, self).__init__()

        self.config = config
        self.seed = random.seed(seed)
        self.state_size=state_size
        self.state_size=action_size

        # self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        # self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        # self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        # self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.actor_local = Actor(state_size, action_size, seed, hidden_layer_param=self.config.ACTOR_H,
                                 output_type=self.config.OUTPUT_TYPE).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, seed,
                                  hidden_layer_param=self.config.ACTOR_H, output_type=self.config.OUTPUT_TYPE).to(DEVICE)
        self.critic_local = Critic(
            state_size, action_size, seed, hidden_layer_param=self.config.CRITIC_H).to(DEVICE)
        self.critic_target = Critic(
            state_size, action_size, seed, hidden_layer_param=self.config.CRITIC_H).to(DEVICE)

        self.actor_opt = optim.Adam(
            self.actor_local.parameters(), lr=self.config.LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic_local.parameters(
        ), lr=self.config.LR_CRITIC, weight_decay=self.config.WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed, self.config.MU,
                             self.config.THETA, self.config.SIGMA)

        # Replay memory
        self.memory = ReplayBuffer(
            action_size, seed, self.config.BATCH_SIZE, self.config.BUFFER_SIZE)

        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        # What the policy is tuned for
        self.output_type = self.config.OUTPUT_TYPE

        self.last_action_probs = np.full(
            shape=action_size,  fill_value=1/action_size,  dtype=np.float)



    def __str__(self):
        print("")
        print("--- Agent Params ---")
        print("Going to train on {}".format(DEVICE))
        print("Learning Rate:: Actor: {} | Critic: {}".format(
            self.config.LR_ACTOR, self.config.LR_CRITIC))
        print("Replay Buffer:: Buffer Size: {} | Sampled Batch size: {}".format(
            self.config.BUFFER_SIZE, self.config.BATCH_SIZE))
        print("")
        print("Actor paramaters:: Input: {} | Hidden Layers: {} | Output: {}".format(
            self.state_size, self.config.ACTOR_H, self.action_size))
        print("Critic paramaters:: Input: {} | Hidden Layers: {} | Output: {}".format(
            self.state_size, [self.config.CRITIC_H[0] + self.action_size, *self.config.CRITIC_H[1:]], 1))
        print(self.actor_local)
        print(self.critic_local)
        print("Output type is: {}".format(self.output_type))
        print("")
        print("")

    def get_title(self):
        for_title = "Network:: A: {} | C: {}\nLearning:: LR_A: {} | LR_C: {} | TAU: {} \nNoise:: MU: {} | THETA: {} | SIGMA: {}\nBuffer:: Size: {} | Batch size: {}".format(self.config.ACTOR_H, self.config.CRITIC_H, self.config.LR_ACTOR, self.config.LR_CRITIC, self.config.TAU, self.config.MU, self.config.THETA, self.config.SIGMA, self.config.BUFFER_SIZE, self.config.BATCH_SIZE)
        for_filename = "A_{} - C_{}".format(' '.join([str(elem) for elem in self.config.ACTOR_H]), ' '.join([str(elem) for elem in self.config.CRITIC_H]))
        return for_title, for_filename


    def reset(self):
        self.noise.reset()
    
    def act(self, state, add_noise=True):
        if type(state) is not np.ndarray:
            state = np.array(state)

        state = torch.from_numpy(state).float().to(DEVICE)
        # action_probs = self.actor_local(state)
        # print(action_probs)
        # selected_index = np.random.choice(len(action_probs), size=1, p=action_probs.detach().numpy())
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()


        if self.output_type is 'probs':
            self.last_action_probs = actions.copy()

            chosen_action = np.random.choice(len(actions), p=actions)
            # print("action_prob: {}".format(actions))
            # print("chosen_action: {}".format(chosen_action))
            return chosen_action

        elif self.output_type is 'vectors':
            if add_noise:
                actions += self.noise.sample()
            return np.clip(actions, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        if self.output_type is 'probs':
            self.memory.add(state, self.last_action_probs, reward, next_state, done)
        elif self.output_type is 'vectors':
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences # these come as tensors
        
        # ---                   Teach Critic (with TD)              --- #
        recommended_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, recommended_actions)
        Q_targets = rewards + (self.config.GAMMA * Q_targets_next * (1 - dones))                 # This is what we actually got from experience
        Q_expected = self.critic_local(states, actions)                       # This is what we thought the expected return of that state-action is.
        critic_loss = F.mse_loss(Q_targets, Q_expected)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_opt.step()


        # ---                   Teach Actor                          --- #
        next_actions = self.actor_local(states)
        # Here we get the value of each state-actions. 
        # This will be backpropagated to the weights that produced the action in the actor network. 
        # Large values will make weights stronger, smaller values (less expected return for that state-action) weaker
        actor_loss = -self.critic_local(states, next_actions).mean()            
        

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()


        # Mix model parameters in both Actor and Critic #
        self.soft_update(self.critic_local, self.critic_target, self.config.TAU) 
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU) 
    
    def soft_update(self, local, target, tau):
        """Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target

            Params
            ======
                local_model: PyTorch model (weights will be copied from)
                target_model: PyTorch model (weights will be copied to)
                tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class OUNoise:
    """Ornstein-Uhlenbeck process. https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process"""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):
        """Initialize parameters and noise process."""
        """ 
            MU is to which the function mean reverts 
            THETA: how "strongly" the system reacts to perturbations
            SIGMA: variation or size of the noise
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
