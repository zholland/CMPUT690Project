import tiles3
from action_value_function import AbstractActionValueFunction
import numpy as np
import random


class FunctionApproximationActionValueFunction(AbstractActionValueFunction):

    def __init__(self, num_dimensions, dimension_ranges, num_actions):
        self.num_tiles = 8192
        self.num_dimensions = num_dimensions
        self.dimension_ranges = dimension_ranges
        self.num_actions = num_actions
        self.theta = [-0.001 * random.random() for _ in range(self.num_tiles*num_actions)]
        self.iht = tiles3.IHT(self.num_tiles)
        self.num_tilings = 16

    def get_inputs(self, S):
        inputs = []
        for i in range(0, self.num_dimensions):
            inputs.append(self.num_tilings * (S[i] / self.dimension_ranges[i]))
        return inputs

    def action_values(self, S):
        sums = np.zeros([self.num_actions])
        if S is not None:
            # Tile code the given inputs.
            phi = tiles3.tiles(self.iht, self.num_tilings, self.get_inputs(S))

            for i in range(0, self.num_tilings):
                for a in range(0, self.num_actions):
                    sums[a] += self.theta[phi[i] + a * self.num_tiles]
        return sums

    def value(self, S, A):
        sum = 0
        if S is not None:
            # Tile code the given inputs.
            phi = tiles3.tiles(self.iht, self.num_tilings, self.get_inputs(S))

            # Sum the weights for all non-zero features.
            for i in range(0, self.num_tilings):
                sum += self.theta[phi[i] + A * self.num_tiles]  # Offset by the action to get to the right segment of theta
        return sum

    def update(self, S, A, new_value):
        # Tile code the given inputs.
        phi = tiles3.tiles(self.iht, self.num_tilings,
                           [self.num_tilings * (S[0] / 1.7), (self.num_tilings * (S[1] / 0.14))])

        # Update theta
        for j in phi:
            self.theta[j + A * self.num_tiles] += new_value