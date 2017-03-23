from n_step_method import NStepMethodBase
import numpy as np


class DQN(NStepMethodBase):

    def __init__(self, env, alpha, epsilon, gamma):
        super().__init__(env, alpha, epsilon, 1, gamma, None)

    def do_learning(self, num_episodes):
        pass

