
from experiments.training import train
import logging
import uuid
import time

from utilities.monitor import save_scores, render_figure, save_states


class Experiment():
    def __init__(self, name, environment, agents, max_t=100, num_episodes=1000, goal = 0., save_states_every = 0, brain_name="", experiment_num=0):
        self.name = name
        self.environment = environment
        self.agents = agents
        self.max_t = max_t
        self.num_episodes = num_episodes
        self.goal = goal
        self.save_states_every = save_states_every

        self.id = uuid.uuid4()
        self.brain_name =brain_name
        self.experiment_num=experiment_num

        logging.basicConfig(filename='experiments/logs/{}-{}.log'.format(time.strftime(
            "%Y-%m-%d_%H%M"),str(self.id)),
                    format='[%(levelname)s]: [%(asctime)s] [%(message)s]', datefmt='%m/%d/%Y %I:%M:%S %p')
        
        self.logger = logging.getLogger(str(self.id))
        self.logger.info("Starting Experiment {}".format(self.name))

    def run(self, development_mode = False):
        self.score_history, self.state_history = [], []
        if development_mode == False:
            try:
                self.score_history, self.state_history = train(env=self.environment,
                                                    agents=self.agents,
                                                    brain_name = self.brain_name,
                                                    max_t=self.max_t,
                                                    num_episodes=self.num_episodes,
                                                    score_history = self.score_history,
                                                    state_history = self.state_history,
                                                    save_states_every=self.save_states_every,
                                                    experiment_num=self.experiment_num)
            except Exception as e:
                print("Encountered an error, going to log into file")
                self.save(self.score_history, self.state_history, display = False, scores_window=100)
                self.__save_error(e)
            finally:
                return self.score_history, self.state_history
        else:
            return train(env=self.environment,
                        agents=self.agents,
                        brain_name = self.brain_name,
                        max_t=self.max_t,
                        num_episodes=self.num_episodes,
                        save_states_every=self.save_states_every,
                        experiment_num=self.experiment_num )



    def save(self, score_history=[], state_history=[], options=['scores', 'figures', 'states'], display = True, scores_window=0):
        if 'scores' in options: save_scores(score_history, agents = self.agents, name = self.name)
        if 'states' in options: save_states(state_history, name=self.name)
        render_figure(score_history, agents = self.agents, name=self.name,  goal=self.goal, display=display, save= 'figures' in options, scores_window=scores_window) 

    def __save_error(self, error):
        self.logger.error(str(error))
