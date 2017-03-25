import tiles3
from action_value_function import AbstractActionValueFunction
import numpy as np
import random


class TileCodingActionValueFunction(AbstractActionValueFunction):

    def __init__(self, num_dimensions, dimension_ranges, num_actions, num_tiles=2048, num_tilings=8, scale_inputs=False):
        self.scale_inputs = scale_inputs
        self.num_tiles = num_tiles
        self.num_dimensions = num_dimensions
        self.dimension_ranges = dimension_ranges
        self.num_actions = num_actions
        self.theta = [-0.001 * random.random() for _ in range(self.num_tiles*num_actions)]
        self.theta = np.asarray(self.theta)
        self.iht = tiles3.IHT(self.num_tiles)
        self.num_tilings = num_tilings

    def get_inputs(self, S):
        inputs = []
        for i in range(0, self.num_dimensions):
            if self.scale_inputs:
                inputs.append(self.num_tilings*(S[i]/self.dimension_ranges[i]))
            else:
                inputs.append(S[i])
        return inputs

    def get_phi(self, S):
        return tiles3.tiles(self.iht, self.num_tilings, self.get_inputs(S))

    def feature_vector(self, S, A):
        psi = np.zeros([self.num_tiles*self.num_actions])
        indicies = tiles3.tiles(self.iht, self.num_tilings, self.get_inputs(S))
        for idx in indicies:
            psi[self.num_tiles*A+idx] = 1
        return psi


    def action_values(self, S):
        sums = np.zeros([self.num_actions])
        if S is not None:
            # Tile code the given inputs.
            phi = self.get_phi(S)

            for i in range(0, self.num_tilings):
                for a in range(0, self.num_actions):
                    sums[a] += self.theta[phi[i] + a * self.num_tiles]
        return sums

    def value(self, S, A):
        sum = 0
        if S is not None:
            # Tile code the given inputs.
            phi = self.get_phi(S)

            # Sum the weights for all non-zero features.
            for i in range(0, self.num_tilings):
                sum += self.theta[phi[i] + A * self.num_tiles]  # Offset by the action to get to the right segment of theta
        return sum

    def update(self, S, A, new_value):
        # Tile code the given inputs.
        phi = self.get_phi(S)

        # Update theta
        for j in phi:
            self.theta[j + A * self.num_tiles] += new_value