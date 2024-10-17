from typing import Dict, Literal
import math

class RewardFunction:
    def __init__(self, ws:Dict, a:float=1.0):
        self.target_name = ws.keys()
        self.ws = ws
        
    def _compute_tanh(self, x):
        return math.tanh(x)

    def _compute_performance_reward(self, x, x_criteria, x_scale, a: float = 3.0):
        xn = x / x_criteria - x_scale
        reward = self._compute_tanh(a * xn)
        return reward

    def _compute_reward(self, state: Dict):

        wdia = state["wdia"]
        dt1 = state["t1"]

        reward_wdia = self._compute_performance_reward(wdia, self.wdia_r, 1, self.a)
        reward_dt = self._compute_performance_reward(dt1, self.dt_r, 1, self.a)

        reward = reward_dt * self.w_dt + reward_wdia * self.w_wdia
        reward /= self.w_dt + self.w_wdia

        return reward

    def _compute_reward_dict(self, state: Dict):

        wdia = state["wdia"]
        dt1 = state["t1"]

        reward_wdia = self._compute_performance_reward(wdia, self.wdia_r, 1, self.a)
        reward_dt = self._compute_performance_reward(dt1, self.dt_r, 1, self.a)

        reward = reward_dt * self.w_dt + reward_wdia * self.w_wdia
        reward /= self.w_dt + self.w_wdia

        reward_dict = {
            "total": reward,
            "wdia": reward_wdia,
            "dt": reward_dt,
        }

        return reward_dict

    def __call__(self, state: Dict):
        reward = self._compute_reward(state)
        return reward
