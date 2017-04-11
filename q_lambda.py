import gym
from gym.envs import register
from joblib import Parallel, delayed
from n_step_method import NStepMethodBase
import numpy as np

from tile_coding_action_value_function import TileCodingActionValueFunction


class Qlearning(NStepMethodBase):
    def __init__(self, env, alpha, epsilon, gamma, action_value_function, epsilon_decay_factor=0.95, lambda_param=0.0):
        super().__init__(env, alpha, epsilon, 1, gamma, action_value_function)
        self.epsilon_decay_factor = epsilon_decay_factor
        self.lambda_param = lambda_param

    def do_learning(self, num_episodes, show_env=False):
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            A = self.epsilon_greedy_action(S)
            done = False
            Rsum = 0
            while not done:
                if show_env:
                    self.env.render()
                Snext, R, done, info = self.env.step(A)
                Rsum += R
                Anext = self.epsilon_greedy_action(Snext)
                self.action_value_function.update(S, A, self.alpha * (
                R + self.gamma * np.max(self.action_value_function.action_values(Snext)) - self.action_value_function.value(S,A)))
                S = Snext
                A = Anext
            self.epsilon *= self.epsilon_decay_factor
            # print(Rsum)
            self.episode_return.append(Rsum)
            if episodeNum >= 100 and np.mean(self.episode_return[episodeNum - 100:episodeNum]) > -110.0:
                break


class RewardsInfo:
    def __init__(self, mean_return):
        self.mean_return = mean_return


def do_experiment(parameters):
    alpha, epsilon = parameters
    env = gym.make('MountainCar-v0')

    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    num_tilings = 8
    action_value_function = TileCodingActionValueFunction(env.observation_space.shape[0],
                                                          dim_ranges,
                                                          env.action_space.n,
                                                          num_tiles=2048,
                                                          num_tilings=num_tilings,
                                                          scale_inputs=True)

    qlearning = Qlearning(env, alpha / num_tilings, epsilon, 1, action_value_function, epsilon_decay_factor=0.98)

    qlearning.do_learning(1000, show_env=False)
    episodes_completed = np.size(qlearning.episode_return)
    print("****alpha: ", alpha, ", epsilon: ",epsilon, "*****")
    print("Mean return: ", np.mean(qlearning.episode_return))
    print("Last 100 Episodes window: ", np.mean(qlearning.episode_return[episodes_completed - 100:episodes_completed]))
    print("Total episodes: ", np.size(qlearning.episode_return))
    print("Total time steps: ", np.abs(np.sum(qlearning.episode_return)))
    return RewardsInfo(np.mean(qlearning.episode_return[episodes_completed - 100:episodes_completed]))


if __name__ == "__main__":
    register(
        id='MountainCar-v3',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000,
        reward_threshold=-110.0,
    )

    rewards_list = []
    # alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
    alphas = [0.4]
    epsilons = [0.1]
    parameters_list = [(alpha, epsilon) for alpha in alphas for epsilon in epsilons]
    for reward_info in Parallel(n_jobs=2)(
            delayed(do_experiment)(parameters) for parameters in parameters_list):
        rewards_list.append(reward_info)
