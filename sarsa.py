import random

from n_step_method import NStepMethodBase


class Sarsa(NStepMethodBase):
    def __init__(self, env, alpha, epsilon, gamma, action_value_function):
        super().__init__(env, alpha, epsilon, 1, gamma, action_value_function)

    def do_learning(self, num_episodes, show_env=True):
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
                # Anext = random.randint(0,2)
                Anext = self.epsilon_greedy_action(Snext)
                self.action_value_function.update(S, A, self.alpha * (
                R + self.gamma * self.action_value_function.value(Snext, Anext) - self.action_value_function.value(S,A)))
                S = Snext
                A = Anext
            # print(Rsum)
            self.episode_return.append(Rsum)
