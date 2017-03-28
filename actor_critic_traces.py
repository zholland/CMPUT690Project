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
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            e_theta = np.zeros(self.parametrizations.theta.shape)
            e_w = np.zeros(self.parametrizations.w.shape)
            I = 1
            done = False
            Rsum = 0
            while not done:
                if show_env:
                    self.env.render()
                A = self.parametrizations.get_action(S)
                Snext, R, done, info = self.env.step(A)
                Rsum += R

                if done:
                    delta = R - self.parametrizations.get_value(S)
                else:
                    delta = R + self.gamma * self.parametrizations.get_value(Snext) - self.parametrizations.get_value(S)

                e_w = self.lambda_w * e_w + I * self.parametrizations.get_phi(S)
                e_theta = self.lambda_theta * e_theta + I * self.parametrizations.get_eligibility_vector(A, S)

                self.parametrizations.w += self.beta * delta * e_w
                self.parametrizations.theta += self.alpha * I * delta * e_theta
                I *= self.gamma
                S = Snext

            print(Rsum)
            self.episode_return.append(Rsum)
            if episodeNum >= 100 and np.mean(self.episode_return[episodeNum - 100:episodeNum]) > 90.0:
                break


if __name__ == "__main__":
    register(
        id='MountainCar-v3',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000,
        reward_threshold=-110.0,
    )

    # env = gym.make('MountainCarContinuous-v0')
    env = gym.make('LunarLander-v2')

    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    num_tilings = 32
    # parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
    #                                                    dim_ranges,
    #                                                    num_tiles=2048,
    #                                                    num_tilings=num_tilings,
    #                                                    scale_inputs=True)

    parametrizations = ActorCriticTileCodingParametrization(env.observation_space.shape[0],
                                                            dim_ranges,
                                                            num_actions=env.action_space.n,
                                                            num_tiles=2 ** 14,
                                                            num_tilings=num_tilings,
                                                            scale_inputs=False)

    alpha = 0.3
    beta = 0.3
    actor_critic = ActorCriticEligibilityTraces(env,
                                                alpha / num_tilings,
                                                beta / num_tilings,
                                                parametrizations,
                                                gamma=0.9,
                                                lambda_theta=0.9,
                                                lambda_w=0.9)

    actor_critic.do_learning(500, show_env=False)
    episodes_completed = np.size(actor_critic.episode_return)
    print("****alpha: ", alpha, ", beta: ", beta, "*****")
    print("Mean return: ", np.mean(actor_critic.episode_return))
    print("Last 100 Episodes window: ",
          np.mean(actor_critic.episode_return[episodes_completed - 100:episodes_completed]))
    print("Total episodes: ", np.size(actor_critic.episode_return))
    print("Total time steps: ", np.abs(np.sum(actor_critic.episode_return)))
