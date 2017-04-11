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
        self.theta = np.zeros([self.num_tiles*2])
        # [0.001 * random.random() for _ in range(self.num_tiles * 2)]
        self.theta[0:num_tiles] = np.asarray([0.0001 * random.random() for _ in range(self.num_tiles)])
        # self.theta[num_tiles:num_tiles*2] = np.asarray([1.0/num_tilings for _ in range(self.num_tiles)])
        self.theta[num_tiles:num_tiles*2] = np.asarray([0 for _ in range(self.num_tiles)])

    def mu_value(self, S):
        return np.dot(self.theta[0:self.num_tiles], self.get_phi(S))

    def sigma_value(self, S):
        # print(np.dot(self.theta[self.num_tiles:self.theta.size], self.get_phi(S)))
        # return 1.0
        return np.exp(min(100, np.dot(self.theta[self.num_tiles:self.theta.size], self.get_phi(S))))
        # return np.abs(np.dot(self.theta[self.num_tiles:self.theta.size], self.get_phi(S)))

    def get_action(self, S):
        pass

    def get_action_continuous(self, S, epsilon):
        # return np.random.normal(loc=self.mu_value(S), scale=self.sigma_value(S)+0.0001+epsilon)
        return np.random.normal(loc=self.mu_value(S), scale=self.sigma_value(S)+0.0001)

    def get_eligibility_vector(self, A, S):
        mu = self.mu_value(S)
        sigma = self.sigma_value(S)

        # print("mu:", mu)
        # print("sigma:", sigma)

        # grad_mu_of_log_pi = ((A - mu)/sigma**2) * self.theta[0:self.num_tiles]
        # grad_sigma_of_log_pi = (((A - mu)**2) / sigma**2 - 1) * self.theta[self.num_tiles:self.theta.size]

        phi = self.get_phi(S)
        grad_mu_of_log_pi = ((A - mu)/(sigma**2+0.001)) * phi
        grad_sigma_of_log_pi = (((A - mu)**2) / (sigma**2+0.001) - 1) * phi

        return np.concatenate((grad_mu_of_log_pi, grad_sigma_of_log_pi))
