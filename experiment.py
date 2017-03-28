import gym
import numpy as np
from gym.envs import register
from joblib import Parallel, delayed
import pickle

from q_learning import Qlearning
from tile_coding_action_value_function import TileCodingActionValueFunction
from true_online_sarsa_lambda import TrueOnlineSarsaLambda

register(
    id='MountainCar-v3',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10000,
    reward_threshold=-110.0,
)

MAX_EPISODES = 1000
NUM_RUNS = 1
TEST_ENV = 'LunarLander-v2'
NUM_TILINGS = 32

NUM_PARLL_JOBS = 4

# ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ALPHAS = [0.6, 0.7, 0.8]
ALPHAS = [0.3, 0.4, 0.5]
EPSILONS = [0.1]
LAMBDA_PARAMS = [0.0, 0.5]
# LAMBDA_PARAMS = [0, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.999]


# LAMBDA_PARAMS = [0.4, 0.5, 0.6]


def make_q_learner(env, alpha, epsilon, gamma, action_value_function, epsilon_decay_factor, lambda_param=None):
    return Qlearning(env,
                     alpha / NUM_TILINGS,
                     epsilon,
                     gamma,
                     action_value_function,
                     epsilon_decay_factor)


def make_sarsa_learner(env, alpha, epsilon, gamma, action_value_function, epsilon_decay_factor, lambda_param):
    return TrueOnlineSarsaLambda(env,
                                 alpha / NUM_TILINGS,
                                 epsilon,
                                 gamma,
                                 action_value_function,
                                 epsilon_decay_factor,
                                 lambda_param)


class RewardsInfo:
    def __init__(self, algorithm_name, alpha, epsilon, lambda_param, mean_return, episodes_window, total_episodes,
                 total_time_steps):
        self.algorithm_name = algorithm_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.episodes_window = episodes_window
        self.total_episodes = total_episodes
        self.total_time_steps = total_time_steps
        self.mean_return = mean_return


def do_experiment(parameters):
    alpha, epsilon, lambda_param = parameters
    env = gym.make(TEST_ENV)
    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    episodes_completed = []
    episode_returns = []
    episodes_windows = []
    total_time_steps_list = []
    print("Running alpha: ", alpha, ", epsilon: ", epsilon, ", lambda: ", lambda_param)

    for i in range(NUM_RUNS):
        env.reset()
        action_value_function = TileCodingActionValueFunction(env.observation_space.shape[0],
                                                              dim_ranges,
                                                              env.action_space.n,
                                                              num_tiles=2**14,
                                                              num_tilings=NUM_TILINGS,
                                                              scale_inputs=False)

        algorithm = TrueOnlineSarsaLambda(env,
                                          alpha / NUM_TILINGS,
                                          epsilon,
                                          1.0,
                                          action_value_function,
                                          epsilon_decay_factor=1.0,
                                          lambda_param=lambda_param)

        # algorithm = Qlearning(env,
        #                       alpha / NUM_TILINGS,
        #                       epsilon,
        #                       1,
        #                       action_value_function,
        #                       epsilon_decay_factor=0.98)

        algorithm.do_learning(MAX_EPISODES, show_env=False)

        episodes_completed.append(np.size(algorithm.episode_return))
        episode_returns.append(np.mean(algorithm.episode_return))
        episodes_windows.append(
            algorithm.episode_return[np.size(algorithm.episode_return) - 100:np.size(algorithm.episode_return)])
        total_time_steps_list.append(np.abs(np.sum(algorithm.episode_return)))
    return RewardsInfo(algorithm_name="t o Sarsa",
                       alpha=alpha,
                       epsilon=epsilon,
                       lambda_param=lambda_param,
                       mean_return=episode_returns,
                       episodes_window=episodes_windows,
                       total_episodes=episodes_completed,
                       total_time_steps=total_time_steps_list)


def run_experiment():
    rewards_list = []
    parameters_list = [(alpha, epsilon, lambda_param) for alpha in ALPHAS for epsilon in EPSILONS for lambda_param in
                       LAMBDA_PARAMS]
    for reward_info in Parallel(n_jobs=NUM_PARLL_JOBS)(
            delayed(do_experiment)(parameters) for parameters in parameters_list):
        rewards_list.append(reward_info)
    pickle.dump(rewards_list, open("lander_rewards_list.p", "wb"))
    return rewards_list


def print_best_results(rewards_list):
    best_reward = None
    for reward_obj in rewards_list:
        if best_reward is None or np.mean(reward_obj.episodes_window) > np.mean(best_reward.episodes_window):
            best_reward = reward_obj

    print("Best algorithm: alpha: ", best_reward.alpha, ", epsilon: ", best_reward.epsilon, ", lambda: ",
          best_reward.lambda_param)
    print("Mean return: ", np.mean(best_reward.mean_return))
    print("Last 100 Episodes window: ", np.mean(best_reward.episodes_window))
    print("Total episodes before solve: ", np.mean(best_reward.total_episodes))
    print("Max episodes before solve: ", np.max(best_reward.total_episodes))
    print("Total time steps: ", np.mean(best_reward.total_time_steps))


def print_all_results(reward_file):
    rewards_list = pickle.load(open(reward_file, "rb"))

    for reward in rewards_list:
        print("Algorithm: alpha: ", reward.alpha, ", epsilon: ", reward.epsilon, ", lambda: ",
              reward.lambda_param)
        print("Mean return: ", np.mean(reward.mean_return))
        print("Last 100 Episodes window: ", np.mean(reward.episodes_window))
        print("Total episodes before solve: ", np.mean(reward.total_episodes))
        print("Max episodes before solve: ", np.max(reward.total_episodes))
        print("Total time steps: ", np.mean(reward.total_time_steps))

if __name__ == "__main__":
    rewards_list = run_experiment()
    print_best_results(rewards_list)
    # print_all_results("/home/zach/PycharmProjects/CMPUT690Project/cart_pole_rewards_list.p")
    # print_best_results(rewards_list = pickle.load(open("/home/zach/PycharmProjects/CMPUT690Project/cart_pole_rewards_list.p", "rb")))

