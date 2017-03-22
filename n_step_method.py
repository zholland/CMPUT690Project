import abc
import numpy as np


"""
Base class for creating n-step TD control methods.
"""
class NStepMethodBase(metaclass=abc.ABCMeta):

    def __init__(self, env, alpha, epsilon, n, gamma, action_value_function):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.gamma = gamma
        self.action_value_function = action_value_function
        self.episode_return = []

    @abc.abstractmethod
    def do_learning(self, num_episodes):
        """"""

    """
    Returns the probability of choosing action A in state S under the epsilon greedy policy.
    """
    def epsilon_greedy_probability(self, A, S):
        greedy_action = np.argmax(self.action_value_function.action_values(S))
        epsilon_fraction = self.epsilon / self.env.action_space.n
        return 1 - self.epsilon + epsilon_fraction if A == greedy_action else epsilon_fraction

    """
    Returns an action selected by the epsilon greedy policy.
    """
    def epsilon_greedy_action(self, S):
        if self.epsilon > np.random.random():
            # Get action randomly
            return self.env.action_space.sample()
        else:
            # Get action greedily according to the action value function
            return np.argmax(self.action_value_function.action_values(S))

    def random_action(self):
        return self.env.action_space.sample()

    def random_action_probability(self):
        return 1 / self.env.action_space

    """
    Returns the value of state S
    """
    def value(self, S):
        value = 0
        for a in range(0, self.env.action_space):
            value += self.epsilon_greedy_probability(a, S) * self.action_value_function.value(S, a)
        return value / self.env.action_space

    def max_action_value(self, S):
        return np.max(self.action_value_function.action_values(S))
