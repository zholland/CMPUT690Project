import gym
import numpy as np
from gym.envs import register
from joblib import Parallel, delayed
import pickle
import matplotlib.pyplot as plt
from q_learning import Qlearning
from tile_coding_action_value_function import TileCodingActionValueFunction
from true_online_sarsa_lambda import TrueOnlineSarsaLambda

def do_demo():
    env = gym.make("CartPole-v0")
    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]
    env.reset()
    action_value_function = TileCodingActionValueFunction(env.observation_space.shape[0],
                                                          dim_ranges,
                                                          env.action_space.n,
                                                          num_tiles=2 ** 11,
                                                          num_tilings=16,
                                                          scale_inputs=False)
    algorithm = TrueOnlineSarsaLambda(env,
                                      alpha=0.7 / 16,
                                      epsilon=1.0,
                                      gamma=1.0,
                                      action_value_function=action_value_function,
                                      epsilon_decay_factor=0.9,
                                      lambda_param=0.5)

    algorithm.do_learning(num_episodes=500, show_env=False, target_reward=195.0, target_window=50)

    # Display result
    algorithm.alpha = 0.0
    algorithm.do_learning(num_episodes=3, show_env=True)


if __name__ == "__main__":
    do_demo()
