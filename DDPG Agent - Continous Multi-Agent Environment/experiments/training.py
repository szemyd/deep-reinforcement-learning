from collections import deque
import numpy as np
from utilities.helper import flatten
import torch

def train(env, agents, brain_name, max_t, num_episodes, score_history=[], state_history=[], scores_window=100, print_every = 20, save_states_every = 0, experiment_num=0):

    scores_deque = deque(score_history[-scores_window:], maxlen=scores_window)
    last_running_mean = float('-inf')

    for episode in range(num_episodes):
        states = env.reset(train_mode=True)[brain_name].vector_observations
        scores = np.zeros(len(agents))

        for i in range(max_t):
            if not save_states_every < 1 and episode % save_states_every == 0:
                state_history.append(states)


            actions = [agent.act(state) for agent, state in zip(agents, states)]

            env_info = env.step(actions)[brain_name]
            next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done  
            [agent.step(state, action, reward, next_state, done) for agent, state, action, reward, next_state, done in zip(agents, states, actions, rewards, next_states, dones)]

            scores += rewards

            states = next_states
            if np.any(dones) == True:
                break


        score_history.append(scores)
        best_in_episode = np.max(scores)
        scores_deque.append(best_in_episode)

        [agent.reset() for agent in agents]
        if episode > scores_window:
            if np.mean(scores_deque) > last_running_mean:
                    # print("")
                    # print('Last {} was better, going to save it'.format(scores_window))
                    for j, agent in enumerate(agents):
                        torch.save(agent.actor_local.state_dict(), 'experiments/exp_{}__agent_{}_actor.pth'.format(experiment_num, j))
                        torch.save(agent.critic_local.state_dict(), 'experiments/exp_{}__agent_{}_critic.pth'.format(experiment_num, j))
                    last_running_mean = np.mean(scores_deque)

                    [agent.save() for agent in agents]
                    last_running_mean = np.mean(scores_deque)
     
            print("\r", 'Total score (max over agents) {} episode: {} | \tAvarage in last {} is {}'.format(episode, best_in_episode, scores_window, np.mean(scores_deque)), end="")


    return score_history, state_history
