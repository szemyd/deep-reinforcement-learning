import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from helper import get_constant_string
from constants import *             # Capital lettered variables are constants from the constants.py file

import os

def calculate_moving_avarage(scores, num_agent=1, scores_window=SCORES_WINDOW):
    single_agent_returns = scores
    moving_avarages=[np.convolve(scores[i], np.ones(scores_window)/scores_window, mode='valid') for i in range(num_agent)]
    
    return moving_avarages


def render_save_graph(scores, save_name=None, scores_window = 0, num_agent = 1, path= SAVE_EXP_PATH, goal=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(np.arange(1, len(scores)+1), scores)

    if scores_window > 0:
        moving_avarages = calculate_moving_avarage([scores], num_agent, scores_window)

        for i_agent in range(len(moving_avarages)):
            plt.plot(np.arange(len(moving_avarages[i_agent]) + scores_window)[scores_window:], moving_avarages[i_agent], 'g-')

    plt.ylabel('Score')
    plt.xlabel('Episode #')

    if goal>0: plt.axhline(y=goal, color='c', linestyle='--')

    hyperparameter_string, for_filename  = get_constant_string()

    plt.title(hyperparameter_string)

    if save_name:
        plt.savefig('{}{}.jpg'.format(path, save_name), bbox_inches='tight')
    else: 
        plt.savefig("{}Figure_{}_{}.jpg".format(path, time.strftime("%Y-%m-%d_%H%M"), for_filename), bbox_inches='tight')
        print("No file name specified to save to")
    
    plt.show()


def save_scores(scores, path=SAVE_EXP_PATH):
    if not os.path.exists(path):
        print("Directory doesn't exist, going to create one first")
        os.makedirs(path)

    _, for_filename  = get_constant_string()

    with open("{}Scores_{}_{}.csv".format(path, time.strftime("%Y-%m-%d_%H%M"), for_filename), 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(scores)

    print("Scores saved!")


def read_scores(network_name=''.format(time.strftime("%Y-%m-%d_%H%M")), path=SAVE_EXP_PATH):

    if os.path.exists(path):

        # _, for_filename  = get_constant_string()

        with open("{}{}.csv".format(path, network_name), newline='') as f:
            reader = csv.reader(f)
            read_score_history = list(reader)[0]

        parsed = [float(i) for i in read_score_history]

        return parsed