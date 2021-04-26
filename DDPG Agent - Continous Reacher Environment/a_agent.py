## Deep Deterministic Policy Gradients ##
from a_model import Actor, Critic    # These are our models
# from model_provided import Actor, Critic    # These are our models
import numpy as np
import random                       # Used for random seed
import copy                         # This is used for the mixing of target and local model parameters

from constants import *             # Capital lettered variables are constants from the constants.py file
from a_memory import ReplayBuffer     # Our replaybuffer, where we store the experiences

import torch
import torch.nn.functional as F
import torch.optim as optim

class DDPG_Agent():
    def __init__(self, state_size, action_size, random_seed, actor_hidden= [400, 300], critic_hidden = [400, 300]):
        super(DDPG_Agent, self).__init__()

        self.seed = random.seed(random_seed)

        # self.actor_local = Actor(state_size, action_size, random_seed).to(DEVICE)
        # self.actor_target = Actor(state_size, action_size, random_seed).to(DEVICE)
        # self.critic_local = Critic(state_size, action_size, random_seed).to(DEVICE)
        # self.critic_target = Critic(state_size, action_size, random_seed).to(DEVICE)
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_layer_param=actor_hidden).to(DEVICE)
        self.actor_target = Actor(state_size, action_size, random_seed, hidden_layer_param=actor_hidden).to(DEVICE)
        self.critic_local = Critic(state_size, action_size, random_seed, hidden_layer_param=critic_hidden).to(DEVICE)
        self.critic_target = Critic(state_size, action_size, random_seed, hidden_layer_param=critic_hidden).to(DEVICE)

        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Noise process
        self.noise = OUNoise(action_size, random_seed, MU, THETA, SIGMA)

        # Replay memory
        self.memory = ReplayBuffer(action_size, random_seed)

        self.soft_update(self.critic_local, self.critic_target, 1) 
        self.soft_update(self.actor_local, self.actor_target, 1) 

        print("")
        print("--- Agent Params ---")
        print("Going to train on {}".format(DEVICE))
        print("Learning Rate:: Actor: {} | Critic: {}".format(LR_ACTOR, LR_CRITIC))
        print("Replay Buffer:: Buffer Size: {} | Sampled Batch size: {}".format(BUFFER_SIZE, BATCH_SIZE))
        print("")
        print("Actor paramaters:: Input: {} | Hidden Layers: {} | Output: {}".format(state_size, actor_hidden, action_size))
        print("Critic paramaters:: Input: {} | Hidden Layers: {} | Output: {}".format(state_size, [critic_hidden[0] + action_size, *critic_hidden[1:]], 1))
        print(self.actor_local)
        print(self.critic_local)
        print("")
        print("")


    def reset(self):
        self.noise.reset()
    
    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(DEVICE)
        # action_probs = self.actor_local(state)
        # print(action_probs)
        # selected_index = np.random.choice(len(action_probs), size=1, p=action_probs.detach().numpy())
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            actions += self.noise.sample()

        return np.clip(actions, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences
 

        # ---                   Teach Critic (with TD)              --- #
        recommended_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, recommended_actions)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))                 # This is what we actually got from experience
        Q_expected = self.critic_local(states, actions)                       # This is what we thought the expected return of that state-action is.
        critic_loss = CRITERION(Q_targets, Q_expected)

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
        self.soft_update(self.critic_local, self.critic_target, TAU) 
        self.soft_update(self.actor_local, self.actor_target, TAU) 
    
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