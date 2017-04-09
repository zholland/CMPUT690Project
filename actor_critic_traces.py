import gym
import numpy as np

from gym.envs import register

from actor_critic_tile_coding_parametrization import ActorCriticTileCodingParametrization
from continuous_actor_critic_tile_coding import ContinuousActorCriticTileCoding


class ActorCriticEligibilityTraces:
    def __init__(self, env, alpha, beta, parametrizations, gamma=1.0, lambda_theta=0.0, lambda_w=0.0):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.parametrizations = parametrizations
        self.lambda_w = lambda_w
        self.lambda_theta = lambda_theta
        self.episode_return = []

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
            while not done:
                if show_env:
                    self.env.render()
                A = self.parametrizations.get_action_continuous(S, epsilon)
                # A = self.parametrizations.get_action(S)
                Snext, R, done, info = self.env.step([A])
                Rsum += R

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

            print(Rsum)
            epsilon *= 0.95
            self.episode_return.append(Rsum)
            if episodeNum >= 100 and np.mean(self.episode_return[episodeNum - 100:episodeNum]) > -110.0:
                break


if __name__ == "__main__":
    register(
        id='MountainCar-v3',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000,
        reward_threshold=-110.0,
    )

    register(
        id='MountainCarContinuous-v1',
        entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
        max_episode_steps=5000,
        reward_threshold=-110.0,
    )

    # env = gym.make('Pendulum-v0')
    env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('LunarLander-v2')

    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    num_tilings = 8
    parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
                                                       dim_ranges,
                                                       num_tiles=2**11,
                                                       num_tilings=num_tilings,
                                                       scale_inputs=True)

    # parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
    #                                                    dim_ranges,
    #                                                    num_tiles=2**14,
    #                                                    num_tilings=num_tilings,
    #                                                    scale_inputs=True)

    # parametrizations = ActorCriticTileCodingParametrization(env.observation_space.shape[0],
    #                                                         dim_ranges,
    #                                                         num_actions=env.action_space.n,
    #                                                         num_tiles=2 ** 11,
    #                                                         num_tilings=num_tilings,
    #                                                         scale_inputs=True)

    ### for continuous ###
    alpha = 0.01
    beta = 1.0
    actor_critic = ActorCriticEligibilityTraces(env,
                                                alpha / num_tilings,
                                                beta / num_tilings,
                                                parametrizations,
                                                gamma=1.0,
                                                lambda_theta=0.5,
                                                lambda_w=0.5)

    ## For pendulum ##
    # alpha = 0.1
    # beta = 0.1
    # actor_critic = ActorCriticEligibilityTraces(env,
    #                                             alpha / num_tilings,
    #                                             beta / num_tilings,
    #                                             parametrizations,
    #                                             gamma=1.0,
    #                                             lambda_theta=0.75,
    #                                             lambda_w=0.75)

    # alpha = 0.3
    # beta = 0.3
    # actor_critic = ActorCriticEligibilityTraces(env,
    #                                             alpha / num_tilings,
    #                                             beta / num_tilings,
    #                                             parametrizations,
    #                                             gamma=1.0,
    #                                             lambda_theta=0.5,
    #                                             lambda_w=0.5)

    actor_critic.do_learning(500, show_env=False)
    episodes_completed = np.size(actor_critic.episode_return)
    print("****alpha: ", alpha, ", beta: ", beta, "*****")
    print("Mean return: ", np.mean(actor_critic.episode_return))
    print("Last 100 Episodes window: ",
          np.mean(actor_critic.episode_return[episodes_completed - 100:episodes_completed]))
    print("Total episodes: ", np.size(actor_critic.episode_return))
    print("Total time steps: ", np.abs(np.sum(actor_critic.episode_return)))
