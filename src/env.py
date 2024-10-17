import numpy as np
from src.simulator import Simulator

class Env:
    def __init__(self, emulator: Simulator, reward_func):
        self.emulator = emulator
        self.reward_func = reward_func

        self.actions = []
        self.states = []
        self.trajs = []
        self.rewards = []

        self.current_reward = None
        self.current_state = None
        self.current_action = None
        self.current_traj = None

        self.init_state = None
        self.init_action = None
        self.init_reward = None
        self.init_traj = None

        # optimization status
        self.optim_status = {}

    def step(self, action):
        
        # simulation with feedfoward control
        result = self.emulator.play(action)

        Ip = result["Ip"]
        betaN = result["betaN"]
        traj = result["traj"]
        
        state = {
            "Ip":Ip,
            "betaN":betaN,
        }

        optim_status = self.reward_func._compute_reward_dict(state)

        # compute reward
        reward = self.reward_func(state)

        # update state and action
        self.current_state = state
        self.current_action = action
        self.current_reward = reward
        self.current_traj = traj
        
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.trajs.append(traj)

        # optimization process status logging
        if optim_status is not None:
            
            for key, value in optim_status.items():
                
                if key not in self.optim_status.keys():
                    self.optim_status[key] = []

                self.optim_status[key].append(value)

        return state, reward, False, {}

    def close(self):
        self.actions.clear()
        self.states.clear()
        self.rewards.clear()
        self.trajs.clear()

        self.current_action = None
        self.current_state = None
        self.current_reward = None
        self.current_traj = None

        self.init_state = None
        self.init_action = None
        self.init_reward = None
        self.init_traj = None

    def reset(self):
        self.current_action = None
        self.current_state = None
        self.current_reward = None
        self.current_traj = None