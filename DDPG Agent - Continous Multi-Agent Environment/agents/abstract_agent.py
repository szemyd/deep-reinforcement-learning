from abc import ABC, abstractmethod
 
class Agent(ABC):
    @abstractmethod

    def reset(self):
        pass

    def act(self, state, eps):
        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def save(self):
        pass

    def get_title(self):
        pass