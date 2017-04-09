import pyximport; pyximport.install(pyimport=True)
import gym
import numpy as np
from tile_coding_action_value_function import TileCodingActionValueFunction
from true_online_sarsa_lambda import TrueOnlineSarsaLambda

env = gym.make('MountainCar-v0')

dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
              range(0, env.observation_space.high.size)]

num_tilings = 8
action_value_function = TileCodingActionValueFunction(env.observation_space.shape[0],
                                                      dim_ranges,
                                                      env.action_space.n,
                                                      num_tiles=2 ** 11,
                                                      num_tilings=num_tilings,
                                                      scale_inputs=True)

epsilon = 0.1
alpha = 0.3
t_o_sarsa_lambda = TrueOnlineSarsaLambda(env,
                                         alpha / num_tilings,
                                         epsilon,
                                         gamma=1.0,
                                         action_value_function=action_value_function,
                                         epsilon_decay_factor=0.99,
                                         lambda_param=0.9)

t_o_sarsa_lambda.do_learning(500, show_env=False)
episodes_completed = np.size(t_o_sarsa_lambda.episode_return)
print("****alpha: ", alpha, ", epsilon: ", epsilon, "*****")
print("Mean return: ", np.mean(t_o_sarsa_lambda.episode_return))
print("Last 100 Episodes window: ",
      np.mean(t_o_sarsa_lambda.episode_return[episodes_completed - 100:episodes_completed]))
print("Total episodes: ", np.size(t_o_sarsa_lambda.episode_return))
print("Total time steps: ", np.abs(np.sum(t_o_sarsa_lambda.episode_return)))