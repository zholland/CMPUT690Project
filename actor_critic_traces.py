import gym
import numpy as np
from joblib import Parallel, delayed
from gym.envs import register
import pickle
from actor_critic_tile_coding_parametrization import ActorCriticTileCodingParametrization
from continuous_actor_critic_tile_coding import ContinuousActorCriticTileCoding


class ActorCriticEligibilityTraces:
    def __init__(self, env, alpha, beta, parametrizations, gamma=1.0, lambda_theta=0.0, lambda_w=0.0, epsilon_decay=1.0):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.parametrizations = parametrizations
        self.lambda_w = lambda_w
        self.lambda_theta = lambda_theta
        self.episode_return = []
        self.epsilon_decay = epsilon_decay
        self.time_steps = []

    def do_learning(self, num_episodes, show_env=False):
        epsilon = 0.0
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            e_theta = np.zeros(self.parametrizations.theta.shape)
            e_w = np.zeros(self.parametrizations.w.shape)
            R_bar = 0.0
            I = 1
            done = False
            Rsum = 0
            t = 0
            while not done:
                if show_env:
                    self.env.render()
                A = min(max(self.parametrizations.get_action_continuous(S, epsilon),-1.0), 1.0)
                # A = self.parametrizations.get_action(S)
                Snext, R, done, info = self.env.step([A])
                Rsum += R
                t += 1

                if done:
                    delta = R - R_bar - self.parametrizations.get_value(S)
                else:
                    delta = R - R_bar + self.gamma * self.parametrizations.get_value(Snext) - self.parametrizations.get_value(S)

                # R_bar += 0.05 * delta
                R_bar += 0.0 * delta

                e_w = self.lambda_w * e_w + I * self.parametrizations.get_phi(S)
                e_theta = self.lambda_theta * e_theta + I * self.parametrizations.get_eligibility_vector(A, S)

                self.parametrizations.w += self.beta * delta * e_w
                # self.parametrizations.theta += self.alpha * parametrizations.sigma_value(S)**2 * I * delta * e_theta
                self.parametrizations.theta += self.alpha * I * delta * e_theta
                I *= self.gamma
                S = Snext

            epsilon *= self.epsilon_decay
            self.episode_return.append(Rsum)
            self.time_steps.append(t)
            if episodeNum >= 10 and np.mean(self.episode_return[episodeNum - 10:episodeNum]) > 90.0:
                break


MAX_EPISODES = 1000
NUM_RUNS = 100
NUM_TILINGS = 8

TEST_ENV = "MountainCarContinuous-v0"

NUM_PARLL_JOBS = 4

# ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# ALPHAS = [0.6, 0.7, 0.8]
# ALPHAS = [0.005]
ALPHAS = [0.005]
BETAS = [1.0]
# ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# ALPHAS = [0.5]
EPSILONS = [0.0]
EPSILONS_DECAY = [1.0]


def tau_to_lambda(tau):
    return 1.0 - (1.0/tau)


# LAMBDA_PARAMS = tau_to_lambda(np.asarray([1, 2, 4, 8, 16, 32]))
# LAMBDA_PARAMS = tau_to_lambda(np.asarray([1, 2, 4]))
LAMBDA_PARAMS = tau_to_lambda(np.asarray([1]))
# LAMBDA_PARAMS = [0.0, 0.5, 0.9]
# LAMBDA_PARAMS = [0, 0.5, 0.75, 0.8, 0.9, 0.95, 0.99, 0.999]
# LAMBDA_PARAMS = [0.4, 0.5, 0.6]

class RewardsInfo:
    def __init__(self, algorithm_name, alpha, beta, epsilon, lambda_param, mean_return, episodes_window, total_episodes,
                 total_time_steps, epsilon_decay):
        self.algorithm_name = algorithm_name
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.lambda_param = lambda_param
        self.epsilon_decay = epsilon_decay
        self.episodes_window = episodes_window
        self.total_episodes = total_episodes
        self.total_time_steps = total_time_steps
        self.mean_return = mean_return


def do_experiment(parameters):
    alpha, beta, epsilon, lambda_param = parameters
    env = gym.make(TEST_ENV)
    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    episodes_completed = []
    episode_returns = []
    episodes_windows = []
    total_time_steps_list = []
    print("Running alpha: ", alpha, ", beta: ", beta, ", epsilon: ", epsilon, ", lambda: ", lambda_param, ", eps_decay: ", 1.0)

    for i in range(NUM_RUNS):
        env.reset()
        parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
                                                       dim_ranges,
                                                       num_tiles=2**11,
                                                       num_tilings=NUM_TILINGS,
                                                       scale_inputs=True)

        algorithm = ActorCriticEligibilityTraces(env,
                                                alpha / NUM_TILINGS,
                                                beta / NUM_TILINGS,
                                                parametrizations,
                                                gamma=1.0,
                                                lambda_theta=lambda_param,
                                                lambda_w=lambda_param)

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
            algorithm.episode_return[np.size(algorithm.episode_return) - 10:np.size(algorithm.episode_return)])
        total_time_steps_list.append(np.sum(algorithm.time_steps))
    return RewardsInfo(algorithm_name="Continuous Actor Critic",
                       alpha=alpha,
                       beta=beta,
                       epsilon=epsilon,
                       lambda_param=lambda_param,
                       mean_return=episode_returns,
                       episodes_window=episodes_windows,
                       total_episodes=episodes_completed,
                       total_time_steps=total_time_steps_list,
                       epsilon_decay=1.0)


def run_experiment():
    rewards_list = []
    parameters_list = [(alpha, beta, epsilon, lambda_param) for alpha in ALPHAS for beta in BETAS for epsilon in EPSILONS for lambda_param in
                       LAMBDA_PARAMS]
    for reward_info in Parallel(n_jobs=NUM_PARLL_JOBS)(
            delayed(do_experiment)(parameters) for parameters in parameters_list):
        rewards_list.append(reward_info)
    pickle.dump(rewards_list, open("continuous_mc_100_runs.p", "wb"))
    return rewards_list


def print_best_results(rewards_list):
    best_reward = None
    for reward_obj in rewards_list:
        # if best_reward is None or np.mean(reward_obj.episodes_window) > np.mean(best_reward.episodes_window):
        #     best_reward = reward_obj
        if best_reward is None or np.mean(reward_obj.total_time_steps) < np.mean(best_reward.total_time_steps) and np.mean(reward_obj.episodes_window) > 90:
            best_reward = reward_obj

    print("Best algorithm: alpha: ", best_reward.alpha, "beta: ", best_reward.beta, ", epsilon: ", best_reward.epsilon, ", lambda: ",
          best_reward.lambda_param, ", eps_decay: ", best_reward.epsilon_decay)
    print("Mean return: ", np.mean(best_reward.mean_return))
    print("Last 10 Episodes window: ", np.mean(best_reward.episodes_window))
    print("Total episodes before solve: ", np.mean(best_reward.total_episodes), "+-", np.std(best_reward.total_episodes) / np.sqrt(len(best_reward.total_episodes)))
    print("Max episodes before solve: ", np.max(best_reward.total_episodes))
    print("Total time steps: ", np.mean(best_reward.total_time_steps), "+-", np.std(best_reward.total_time_steps) / np.sqrt(len(best_reward.total_time_steps)))


def print_all_results(reward_file):
    rewards_list = pickle.load(open(reward_file, "rb"))

    for reward in rewards_list:
        if np.mean(reward.episodes_window) > 90.0:
            print("Algorithm: alpha: ", reward.alpha, "beta: ", reward.beta, ", epsilon: ", reward.epsilon, ", lambda: ",
                  reward.lambda_param, ", eps_decay: ", reward.epsilon_decay)
            print("Mean return: ", np.mean(reward.mean_return))
            print("Last 10 Episodes window: ", np.mean(reward.episodes_window))
            print("Total episodes before solve: ", np.mean(reward.total_episodes), "+-", np.std(reward.total_episodes) / np.sqrt(len(reward.total_episodes)))
            print("Max episodes before solve: ", np.max(reward.total_episodes))
            print("Total time steps: ", np.mean(reward.total_time_steps), "+-", np.std(reward.total_time_steps) / np.sqrt(len(reward.total_time_steps)))

if __name__ == "__main__":
    rewards_list = run_experiment()
    print_best_results(rewards_list)
    # print_all_results("/home/zach/PycharmProjects/CMPUT690Project/continuous_mc.p")
    # print_best_results(rewards_list=pickle.load(open("/home/zach/PycharmProjects/CMPUT690Project/continuous_mc.p", "rb")))
    # register(
    #     id='MountainCar-v3',
    #     entry_point='gym.envs.classic_control:MountainCarEnv',
    #     max_episode_steps=10000,
    #     reward_threshold=-110.0,
    # )
    #
    # register(
    #     id='MountainCarContinuous-v1',
    #     entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    #     max_episode_steps=5000,
    #     reward_threshold=-110.0,
    # )
    #
    # # env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # # env = gym.make('LunarLander-v2')
    #
    # dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
    #               range(0, env.observation_space.high.size)]
    #
    # num_tilings = 8
    # parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
    #                                                    dim_ranges,
    #                                                    num_tiles=2**11,
    #                                                    num_tilings=num_tilings,
    #                                                    scale_inputs=True)
    #
    # # parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
    # #                                                    dim_ranges,
    # #                                                    num_tiles=2**14,
    # #                                                    num_tilings=num_tilings,
    # #                                                    scale_inputs=True)
    #
    # # parametrizations = ActorCriticTileCodingParametrization(env.observation_space.shape[0],
    # #                                                         dim_ranges,
    # #                                                         num_actions=env.action_space.n,
    # #                                                         num_tiles=2 ** 11,
    # #                                                         num_tilings=num_tilings,
    # #                                                         scale_inputs=True)
    #
    # ### for continuous ###
    # alpha = 0.01
    # beta = 1.0
    # actor_critic = ActorCriticEligibilityTraces(env,
    #                                             alpha / num_tilings,
    #                                             beta / num_tilings,
    #                                             parametrizations,
    #                                             gamma=1.0,
    #                                             lambda_theta=0.5,
    #                                             lambda_w=0.5)
    #
    # ## For pendulum ##
    # # alpha = 0.1
    # # beta = 0.1
    # # actor_critic = ActorCriticEligibilityTraces(env,
    # #                                             alpha / num_tilings,
    # #                                             beta / num_tilings,
    # #                                             parametrizations,
    # #                                             gamma=1.0,
    # #                                             lambda_theta=0.75,
    # #                                             lambda_w=0.75)
    #
    # # alpha = 0.3
    # # beta = 0.3
    # # actor_critic = ActorCriticEligibilityTraces(env,
    # #                                             alpha / num_tilings,
    # #                                             beta / num_tilings,
    # #                                             parametrizations,
    # #                                             gamma=1.0,
    # #                                             lambda_theta=0.5,
    # #                                             lambda_w=0.5)
    #
    # actor_critic.do_learning(500, show_env=False)
    # episodes_completed = np.size(actor_critic.episode_return)
    # print("****alpha: ", alpha, ", beta: ", beta, "*****")
    # print("Mean return: ", np.mean(actor_critic.episode_return))
    # print("Last 100 Episodes window: ",
    #       np.mean(actor_critic.episode_return[episodes_completed - 100:episodes_completed]))
    # print("Total episodes: ", np.size(actor_critic.episode_return))
    # print("Total time steps: ", np.abs(np.sum(actor_critic.episode_return)))
