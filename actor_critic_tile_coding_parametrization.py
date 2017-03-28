import tiles3
import numpy as np
import random


def softindmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()


class ActorCriticTileCodingParametrization:

    def __init__(self, num_dimensions, dimension_ranges, num_actions, num_tiles=2048, num_tilings=8, scale_inputs=False):
        self.scale_inputs = scale_inputs
        self.num_tiles = num_tiles
        self.num_dimensions = num_dimensions
        self.dimension_ranges = dimension_ranges
        self.num_actions = num_actions
        self.iht = tiles3.IHT(self.num_tiles)
        self.num_tilings = num_tilings
        self.w = [-0.001 * random.random() for _ in range(self.num_tiles)]
        self.w = np.asarray(self.w)
        self.theta = [-0.001 * random.random() for _ in range(self.num_tiles*num_actions)]
        self.theta = np.asarray(self.theta)

    def get_inputs(self, S):
        inputs = []
        for i in range(0, self.num_dimensions):
            if self.scale_inputs:
                inputs.append(self.num_tilings*(S[i]/self.dimension_ranges[i]))
            else:
                inputs.append(S[i])
        return inputs

    def get_phi(self, S, A=None):
        indicies = tiles3.tiles(self.iht, self.num_tilings, self.get_inputs(S))

        if A is None:
            phi = np.zeros([self.num_tiles])
            for idx in indicies:
                phi[idx] = 1
            return phi
        else:
            phi = np.zeros([self.num_tiles * self.num_actions])
            for idx in indicies:
                phi[self.num_tiles * A + idx] = 1
            return phi

    def get_action(self, S):
        h_values = np.zeros([self.num_actions])
        for a in range(self.num_actions):
            h_values[a] = np.dot(self.theta, self.get_phi(S, a))
        return np.argmax(softindmax(h_values))

    def get_value(self, S):
        return np.dot(self.get_phi(S), self.w)

    def get_eligibility_vector(self, A, S):
        h_values = np.zeros([self.num_actions])
        for a in range(self.num_actions):
            h_values[a] = np.dot(self.theta, self.get_phi(S, a))
        pi = softindmax(h_values)

        sum = np.zeros([self.num_tiles * self.num_actions])
        for b in range(self.num_actions):
            sum += pi[b] * self.get_phi(S, b)

        return self.get_phi(S, A) - sum
