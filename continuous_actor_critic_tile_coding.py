import random
import numpy as np

from actor_critic_tile_coding_parametrization import ActorCriticTileCodingParametrization


class ContinuousActorCriticTileCoding(ActorCriticTileCodingParametrization):
    def __init__(self, num_dimensions, dimension_ranges, num_tiles=2048, num_tilings=8, scale_inputs=False):
        super().__init__(num_dimensions,
                         dimension_ranges,
                         num_actions=1,
                         num_tiles=num_tiles,
                         num_tilings=num_tilings,
                         scale_inputs=scale_inputs)
        self.theta = [0 * random.random() for _ in range(self.num_tiles * 2)]
        self.theta = np.asarray(self.theta)

    def mu_value(self, S):
        return np.dot(self.theta[0:self.num_tiles], self.get_phi(S))

    def sigma_value(self, S):
        # print(np.dot(self.theta[self.num_tiles:self.theta.size], self.get_phi(S)))
        return np.exp(np.dot(self.theta[self.num_tiles:self.theta.size], self.get_phi(S)))

    def get_action(self, S):
        return np.random.normal(loc=self.mu_value(S), scale=self.sigma_value(S))

    def get_eligibility_vector(self, A, S):
        mu = self.mu_value(S)
        sigma = self.sigma_value(S)

        grad_mu_of_log_pi = (1/sigma**2) * (A - mu) * self.theta[0:self.num_tiles]
        grad_sigma_of_log_pi = (((A - mu)**2) / sigma**2 - 1) * self.theta[self.num_tiles:self.theta.size]

        return np.concatenate((grad_mu_of_log_pi, grad_sigma_of_log_pi))
