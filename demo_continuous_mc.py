import gym
import numpy as np

from actor_critic_traces import ActorCriticEligibilityTraces
from continuous_actor_critic_tile_coding import ContinuousActorCriticTileCoding

NUM_TILINGS = 8

def do_demo():
    env = gym.make("MountainCarContinuous-v0")
    dim_ranges = [env.observation_space.high[i] - env.observation_space.low[i] for i in
                  range(0, env.observation_space.high.size)]
    env.reset()
    parametrizations = ContinuousActorCriticTileCoding(env.observation_space.shape[0],
                                                       dim_ranges,
                                                       num_tiles=2 ** 11,
                                                       num_tilings=NUM_TILINGS,
                                                       scale_inputs=True)

    algorithm = ActorCriticEligibilityTraces(env,
                                             0.005 / NUM_TILINGS,
                                             # 0.05 / NUM_TILINGS,
                                             1.0 / NUM_TILINGS,
                                             parametrizations,
                                             gamma=1.0,
                                             lambda_theta=0.0,
                                             lambda_w=0.0,
                                             epsilon_decay=1.0)

    algorithm.do_learning(num_episodes=500, show_env=False, target_reward=90.0, target_window=10)
    print(np.mean(algorithm.episode_return[np.size(algorithm.episode_return)-10:np.size(algorithm.episode_return)]))
    # Display result
    algorithm.alpha = 0.0
    algorithm.do_learning(num_episodes=2, show_env=True)


if __name__ == "__main__":
    do_demo()
