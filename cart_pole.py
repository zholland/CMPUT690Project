import numpy as np
import gym
from sarsa import Sarsa
from function_approximation_action_value_function import FunctionApproximationActionValueFunction
env = gym.make('CartPole-v0')

# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
print(env.observation_space.high)
print(env.observation_space.low)
dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in range(0, env.observation_space.high.size)]
dim_ranges[1] = 100
dim_ranges[3] = 100
Q = FunctionApproximationActionValueFunction(4, dim_ranges, 2)
sarsa = Sarsa(env, 0.3, 0.2, 1, Q)
sarsa.do_learning(1000, show_env=False)
print(np.mean(sarsa.episode_return))

