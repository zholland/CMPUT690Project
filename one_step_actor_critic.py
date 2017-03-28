import gym
import numpy as np

from gym.envs import register

from actor_critic_tile_coding_parametrization import ActorCriticTileCodingParametrization


class OneStepActorCritic:
    def __init__(self, env, alpha, beta, parametrizations, gamma=1):
        self.env = env
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.parametrizations = parametrizations
        self.episode_return = []

    def do_learning(self, num_episodes, show_env=False):
        for episodeNum in range(num_episodes):
            S = self.env.reset()
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
                self.parametrizations.w += self.beta * delta * self.parametrizations.get_phi(S)
                self.parametrizations.theta += self.alpha * I * delta * self.parametrizations.get_eligibility_vector(A,
                                                                                                                     S)
                I *= self.gamma
                S = Snext

            print(Rsum)
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

    env = gym.make('MountainCar-v3')

    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]

    num_tilings = 8
    parametrizations = ActorCriticTileCodingParametrization(env.observation_space.shape[0],
                                                            dim_ranges,
                                                            env.action_space.n,
                                                            num_tiles=2048,
                                                            num_tilings=num_tilings,
                                                            scale_inputs=True)

    alpha = 0.3
    beta = 0.3
    t_o_sarsa_lambda = OneStepActorCritic(env,
                                          alpha / num_tilings,
                                          beta / num_tilings,
                                          parametrizations,
                                          1)

    t_o_sarsa_lambda.do_learning(500, show_env=False)
    episodes_completed = np.size(t_o_sarsa_lambda.episode_return)
    print("****alpha: ", alpha, ", beta: ", beta, "*****")
    print("Mean return: ", np.mean(t_o_sarsa_lambda.episode_return))
    print("Last 100 Episodes window: ",
          np.mean(t_o_sarsa_lambda.episode_return[episodes_completed - 100:episodes_completed]))
    print("Total episodes: ", np.size(t_o_sarsa_lambda.episode_return))
    print("Total time steps: ", np.abs(np.sum(t_o_sarsa_lambda.episode_return)))
