import numpy as np
import gym
import abc
from gym.envs.registration import register
from sarsa import Sarsa
from q_sigma import QSigma
from function_approximation_action_value_function import FunctionApproximationActionValueFunction

register(
    id='MountainCar-v3',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)
env = gym.make('MountainCar-v3')
# env = gym.make('CartPole-v0')
# env = gym.make('Pendulum-v0')

# for i_episode in range(2):
#     observation = env.reset()
#     for t in range(10001):
#         # env.render()
#         # print(observation)
#         action = env.action_space.sample()
#         # print(action)
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# print(env.observation_space.high)
# print(env.observation_space.low)
dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
              range(0, env.observation_space.high.size)]


# dim_ranges = [0, 0, 0, 0]
# # Cart Position
# dim_ranges[0] = 4.8
# # Cart Velocity
# dim_ranges[1] = 1
# # Pole angle
# dim_ranges[2] = 41.8
# dim_ranges[3] = 1

class AbsractSigma(metaclass=abc.ABCMeta):
    def __init__(self, val):
        self.val = val

    def value(self, episodeNum):
        """"""

    def __hash__(self):
        return hash((type(self).__name__, self.val))

    def __eq__(self, other):
        return (type(self).__name__, self.val) == (type(other).__name__, other.val)


class VariableSigma(AbsractSigma):
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes
        super().__init__(val=-1)

    def value(self, episodeNum):
        return math.exp(-0.05 * episodeNum)


class FixedSigma(AbsractSigma):
    def value(self, episodeNum):
        return self.val


num_episodes = 10000

Q = FunctionApproximationActionValueFunction(env.observation_space.shape[0], dim_ranges, env.action_space.n,
                                             num_tiles=2048, num_tilings=8, scale_inputs=True)
qsigma = QSigma(env, 0.3 / Q.num_tilings, 0.0, 8, 1, Q, FixedSigma(0))
qsigma.do_learning(num_episodes, show_env=False)
print("Mean: ", np.mean(qsigma.episode_return))
episodes_completed = np.size(qsigma.episode_return)
print("Last 100 Episodes window: ", np.mean(qsigma.episode_return[episodes_completed - 100:episodes_completed]))
print("Total episodes: ", np.size(qsigma.episode_return))
print("Total time steps: ", -1*np.sum(qsigma.episode_return))
# qsigma.alpha = 0
# qsigma.epsilon = 0
# qsigma.do_learning(10, show_env=True)
# print("Last 10 Episodes window: ", np.mean(qsigma.episode_return[num_episodes+10 - 10:num_episodes+10]))

