import random
import gym
from n_step_method import NStepMethodBase
import numpy as np
import tensorflow as tf
from deep_nn_action_value_function import Transition
from deep_nn_action_value_function import DeepNNActionValueFunction
from gym.envs.registration import register


class DQN(NStepMethodBase):
    def __init__(self,
                 env,
                 alpha,
                 epsilon=0.0,
                 epsilon_decay_rate=1.0,
                 gamma=1,
                 reset_interval=1,
                 replay_mem_capacity=1000,
                 train_frequency=1,
                 replay_start_size=32,
                 update_batch_size=32,
                 learning_rate=0.001):
        super().__init__(env, alpha, epsilon, 1, gamma, None)
        self.reset_interval = reset_interval
        tf.reset_default_graph()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.sess = tf.Session()
        self.action_value_function = DeepNNActionValueFunction(self.sess, env.observation_space.shape[0],env.action_space.n,gamma, update_batch_size, learning_rate=learning_rate)
        self.sess.run(tf.global_variables_initializer())
        self.action_value_function.copy_estimator_to_target()
        self.replay_mem_capacity = replay_mem_capacity
        self.replay_mem = np.ndarray([replay_mem_capacity], dtype=Transition)
        self.transition_count = 0
        self.train_frequency = train_frequency
        self.replay_start_size = replay_start_size
        self.update_batch_size = update_batch_size
        self.epsilon_decay_rate = epsilon_decay_rate

    def do_learning(self, num_episodes, show_env=False):
        for episodeNum in range(num_episodes):
            S = self.env.reset()
            A = self.epsilon_greedy_action(S)
            done = False
            Rsum = 0
            while not done:
                if show_env:
                    self.env.render()
                Snext, R, done, info = self.env.step(A)
                Rsum += R
                Anext = self.epsilon_greedy_action(Snext)

                # Store transition
                self.replay_mem[self.transition_count % self.replay_mem_capacity] = Transition(S, A, R, Snext, done)
                self.transition_count += 1

                if self.transition_count >= self.replay_start_size and self.transition_count % self.train_frequency == 0:
                    sampled_transitions = np.random.choice(
                        self.replay_mem[0:min(self.transition_count, self.replay_mem_capacity)], size=self.update_batch_size)
                    self.action_value_function.update_batch(sampled_transitions)

                if self.transition_count % self.reset_interval == 0:
                    self.action_value_function.copy_estimator_to_target()

                S = Snext
                A = Anext
            self.epsilon *= self.epsilon_decay_rate
            print(Rsum)
            self.episode_return.append(Rsum)
            if episodeNum >= 100 and np.mean(self.episode_return[episodeNum - 100:episodeNum]) > -110:
                break


if __name__ == "__main__":
    register(
        id='MountainCar-v3',
        entry_point='gym.envs.classic_control:MountainCarEnv',
        max_episode_steps=10000,
        reward_threshold=-110.0,
    )
    env = gym.make('MountainCar-v0')
    # env = gym.make('CartPole-v0')
    ## DQN for cart pole
    # dqn = DQN(
    #     env=env,
    #     alpha=None,
    #     epsilon=1,
    #     epsilon_decay_rate=0.98,
    #     gamma=1,
    #     reset_interval=16,
    #     replay_mem_capacity=100000,
    #     train_frequency=2,
    #     replay_start_size=16,
    #     update_batch_size=16,
    #     learning_rate=0.01
    # )
    ## DQN for mountain car
    dqn = DQN(
            env=env,
            alpha=None,
            epsilon=0.0,
            epsilon_decay_rate=0.98,
            gamma=1,
            reset_interval=1024,
            replay_mem_capacity=100000,
            train_frequency=4,
            replay_start_size=1024,
            update_batch_size=8,
            learning_rate=0.1
    )
    dqn.do_learning(500)
    episodes_completed = np.size(dqn.episode_return)
    print("Mean return: ", np.mean(dqn.episode_return))
    print("Last 100 Episodes window: ", np.mean(dqn.episode_return[episodes_completed - 100:episodes_completed]))
    print("Total episodes: ", np.size(dqn.episode_return))
    print("Total time steps: ", np.sum(dqn.episode_return))
