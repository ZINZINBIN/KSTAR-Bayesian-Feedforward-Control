import numpy as np

class Controller:
    def __init__(self, optimizer, signals):
        self.optimizer = optimizer
    
    def __call__(self, state:np.array):
        return self._action(state)
    
    def _action(self, state:np.array):
        action = self.optimizer.suggest(state)
        return action