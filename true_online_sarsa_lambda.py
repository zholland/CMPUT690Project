import gym
from gym.envs import register

from n_step_method import NStepMethodBase
import numpy as np

from tile_coding_action_value_function import TileCodingActionValueFunction


class TrueOnlineSarsaLambda(NStepMethodBase):
    def __init__(self, env, alpha, epsilon, gamma, action_value_function, epsilon_decay_factor=0.95, lambda_param=0.0):
        super().__init__(env, alpha, epsilon, 1, gamma, action_value_function)
        self.epsilon_decay_factor = epsilon_decay_factor
        self.lambda_param = lambda_param

    def do_learning(self, num_episodes, show_env=False, target_reward=-110.0, target_window=10):
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            A = self.epsilon_greedy_action(S)
            done = False
            Rsum = 0
            psi = self.action_value_function.feature_vector(S, A)
            e = np.zeros([self.action_value_function.num_tiles*self.env.action_space.n])
            Q_old = 0
            while not done:
                if show_env:
                    self.env.render()
                Snext, R, done, info = self.env.step(A)
                Rsum += R
                Anext = self.epsilon_greedy_action(Snext)

                psi_prime = self.action_value_function.feature_vector(Snext, Anext)

                Q = np.dot(psi, self.action_value_function.theta)
                Q_prime = np.dot(psi_prime, self.action_value_function.theta)
                delta = R + self.gamma * Q_prime - Q
                e = self.gamma * self.lambda_param * e + psi - self.alpha * self.gamma * self.lambda_param * np.dot(e, psi) * psi

                self.action_value_function.theta = self.action_value_function.theta + self.alpha * (delta + Q - Q_old) * e - self.alpha * (Q - Q_old) * psi
                Q_old = Q_prime
                psi = psi_prime
                A = Anext

            self.epsilon *= self.epsilon_decay_factor
            # print(Rsum)
            self.episode_return.append(Rsum)
            if episodeNum >= target_window and np.mean(self.episode_return[episodeNum - target_window:episodeNum]) > target_reward:
                break

if __name__ == "__main__":
    register(
        id='MountainCar-v3',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000,
        reward_threshold=-110.0,
    )

    env = gym.make('MountainCar-v0')
    # env = gym.make('Acrobot-v1')

    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    num_tilings = 8
    action_value_function = TileCodingActionValueFunction(env.observation_space.shape[0],
                                                          dim_ranges,
                                                          env.action_space.n,
                                                          num_tiles=2**11,
                                                          num_tilings=num_tilings,
                                                          scale_inputs=True)
    ## For mountain car
    epsilon = 0.0
    alpha = 0.9
    t_o_sarsa_lambda = TrueOnlineSarsaLambda(env,
                                             alpha / num_tilings,
                                             epsilon,
                                             gamma=1.0,
                                             action_value_function=action_value_function,
                                             epsilon_decay_factor=1.0,
                                             lambda_param=0.875)
    #

    # epsilon = 0.2
    # alpha = 0.3
    # t_o_sarsa_lambda = TrueOnlineSarsaLambda(env,
    #                                          alpha / num_tilings,
    #                                          epsilon,
    #                                          gamma=1.0,
    #                                          action_value_function=action_value_function,
    #                                          epsilon_decay_factor=0.98,
    #                                          lambda_param=0.9)

    t_o_sarsa_lambda.do_learning(1000, show_env=False)
    episodes_completed = np.size(t_o_sarsa_lambda.episode_return)
    print("****alpha: ", alpha, ", epsilon: ", epsilon, "*****")
    print("Mean return: ", np.mean(t_o_sarsa_lambda.episode_return))
    print("Last 10 Episodes window: ", np.mean(t_o_sarsa_lambda.episode_return[episodes_completed - 10:episodes_completed]))
    print("Total episodes: ", np.size(t_o_sarsa_lambda.episode_return))
    print("Total time steps: ", np.abs(np.sum(t_o_sarsa_lambda.episode_return)))
    t_o_sarsa_lambda.alpha = 0.0
    t_o_sarsa_lambda.do_learning(10, show_env=True)
