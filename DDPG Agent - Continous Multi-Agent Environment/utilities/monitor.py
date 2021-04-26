import numpy as np
import matplotlib.pyplot as plt
import csv
import time
#from .helper import get_constant_string
# from constants import *             # Capital lettered variables are constants from the constants.py file

import os


def calculate_moving_avarage(scores, num_agent=1, scores_window=100):
    single_agent_returns = np.transpose(np.array(scores))
    moving_avarages = [np.convolve(single_agent_returns[i], np.ones(scores_window)/scores_window, mode='valid') for i in range(num_agent)]

    return moving_avarages

def calculate_max(scores):
    new_scores = scores.copy()

    for i, episode_score in enumerate(new_scores):
        new_scores[i] = np.delete(episode_score, np.argmin(episode_score))

    print(new_scores[-5:])
    return new_scores

def render_figure(scores, agents, name="", scores_window=0, path="", goal=0, save=False, display= True):
    if len(path) < 1:
        path = 'experiments/saved/'

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # --- Plot labels --- #
    parameter_string, for_filename = agents[0].get_title()

    # plt.suptitle(name,fontsize=24, y=1)
    # plt.title(parameter_string,fontsize=16)
    plt.title(parameter_string)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    # --- Plot scores --- #
    if len(agents)>1: # multiple agents
        accumulated_by_agent = np.transpose(np.array(scores))
        for i_agent in range(len(agents)):
            plt.plot(np.arange(1, len(accumulated_by_agent[i_agent])+1), accumulated_by_agent[i_agent])
    else: plt.plot(np.arange(1, len(scores)+1), scores)

    # --- Plot moving avarages --- #
    if scores_window > 0:
        moving_avarages = calculate_moving_avarage(scores, len(agents), scores_window=scores_window)
        best_of_two = calculate_moving_avarage(calculate_max(scores), 1, scores_window=scores_window)

        for i_agent in range(len(moving_avarages)):
            plt.plot(np.arange(len(moving_avarages[i_agent]) + scores_window)[scores_window:], moving_avarages[i_agent], 'g-')
        
        plt.plot(np.arange(len(best_of_two[0]) + scores_window)[scores_window:], best_of_two[0], 'k-')
    if goal > 0.: plt.axhline(y=goal, color='c', linestyle='--')

    # --- Save and Display --- #
    if save: plt.savefig("{}{}_figure_{}.jpg".format(path, time.strftime("%Y-%m-%d_%H%M%S"), name), bbox_inches='tight')
    if display: plt.show()




def save_scores(scores, agents, name="",  path=""):
    if len(path) < 1:
        path = 'experiments/saved/'

    if not os.path.exists(path):
        print("Directory doesn't exist, going to create one first")
        os.makedirs(path)

    _, for_filename = agents[0].get_title()

    with open("{}{}_scores_{}.csv".format(path, time.strftime("%Y-%m-%d_%H%M%S"), name), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(scores)

    print("Scores saved!")


def read_scores(network_name=''.format(time.strftime("%Y-%m-%d_%H%M")), path=''):
    if len(path) < 1:
        path = 'experiments/saved/'

    if os.path.exists(path):

        # _, for_filename  = get_constant_string()

        with open("{}{}.csv".format(path, network_name), newline='') as f:
            reader = csv.reader(f)
            read_score_history = list(reader)[0]

        parsed = [float(i) for i in read_score_history]

        return parsed

def save_states(states, name="", path=""):
    if len(path) < 1:
        path = 'experiments/saved/'

    with open("{}{}_states_{}.csv".format(path, time.strftime("%Y-%m-%d_%H%M%S"), name), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(states)