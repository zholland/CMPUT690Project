from action_value_function import AbstractActionValueFunction
import numpy as np
import tensorflow as tf
import random


class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class QEstimator:
    def __init__(self, state_dims, num_actions, update_batch_size, scope="estimator", learning_rate=0.001):
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.update_batch_size = update_batch_size
        self.scope = scope
        # self.session = tf.InteractiveSession()
        with tf.variable_scope(scope):
            self._create_model_2(learning_rate, hidden_units_per_layer=128)

    def _create_model_2(self, learning_rate=0.001, hidden_units_per_layer=4):
        # Input
        self.x = tf.placeholder(tf.float32, [None, self.state_dims])
        self.a = tf.placeholder(tf.int32, [None, 1])

        # Fully connected
        fc_layer_1 = tf.contrib.layers.fully_connected(inputs=self.x, num_outputs=hidden_units_per_layer, activation_fn=tf.nn.relu)
        fc_layer_2 = tf.contrib.layers.fully_connected(inputs=fc_layer_1, num_outputs=hidden_units_per_layer, activation_fn=tf.nn.relu)
        self.y = tf.contrib.layers.fully_connected(inputs=fc_layer_2, num_outputs=self.num_actions, activation_fn=None)

        # Get the predictions for the chosen actions only
        gather_indices = tf.reshape(tf.range(self.update_batch_size) * self.num_actions,
                                    [self.update_batch_size, 1]) + self.a
        self.a_value_estimates = tf.reshape(
            tf.gather(tf.reshape(self.y, [self.update_batch_size * self.num_actions, 1]), gather_indices),
            [self.update_batch_size, 1])

        # Target placeholder
        self.y_ = tf.placeholder(tf.float32, [None, 1])

        # Loss and optimizer
        loss = tf.reduce_mean(tf.squared_difference(self.y_, self.a_value_estimates))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss, global_step=tf.contrib.framework.get_global_step())
        # self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=tf.contrib.framework.get_global_step())

    def _create_model(self, learning_rate=0.001):
        # Input
        self.x = tf.placeholder(tf.float32, [None, self.state_dims])
        self.a = tf.placeholder(tf.int32, [None, 1])

        # Fully connected
        W_1 = tf.Variable(tf.truncated_normal(shape=[self.state_dims, 64], stddev=0.1))
        b_1 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_1 = tf.nn.relu(tf.matmul(self.x, W_1) + b_1)

        # Fully connected
        W_2 = tf.Variable(tf.truncated_normal(shape=[64, 64], stddev=0.1))
        b_2 = tf.Variable(tf.constant(0.1, shape=[64]))
        h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

        # Fully connected linear output
        W_3 = tf.Variable(tf.truncated_normal(shape=[64, self.num_actions], stddev=0.1))
        b_3 = tf.Variable(tf.constant(0.1, shape=[self.num_actions]))
        self.y = tf.matmul(h_2, W_3) + b_3

        # Get the predictions for the chosen actions only
        gather_indices = tf.reshape(tf.range(self.update_batch_size)*self.num_actions, [self.update_batch_size, 1]) + self.a
        self.a_value_estimates = tf.reshape(tf.gather(tf.reshape(self.y, [self.update_batch_size*self.num_actions, 1]), gather_indices), [self.update_batch_size, 1])

        # Target placeholder
        self.y_ = tf.placeholder(tf.float32, [None, 1])

        # Loss and optimizer
        loss = tf.reduce_mean(tf.squared_difference(self.y_, self.a_value_estimates))
        # self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss, global_step=tf.contrib.framework.get_global_step())
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=tf.contrib.framework.get_global_step())


class DeepNNActionValueFunction(AbstractActionValueFunction):
    def __init__(self, sess, state_dims, num_actions, gamma, update_batch_size, learning_rate=0.001):
        self.sess = sess
        self.gamma = gamma
        self.state_dims = state_dims
        self.num_actions = num_actions
        self.update_batch_size = update_batch_size
        self.estimator = QEstimator(state_dims, num_actions, update_batch_size, scope="estimator", learning_rate=learning_rate)
        self.target = QEstimator(state_dims, num_actions, update_batch_size, scope="target", learning_rate=learning_rate)

    def copy_estimator_to_target(self):
        estimator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

        assign_ops = []
        for estimator_v, target_v in zip(estimator_vars, target_vars):
            op = target_v.assign(estimator_v)
            assign_ops.append(op)

        self.sess.run(assign_ops)

    def action_values(self, state):
        return self.estimator.y.eval(session=self.sess, feed_dict={self.estimator.x: state.reshape(1, self.state_dims)})[0]

    def target_action_values(self, state):
        return self.target.y.eval(session=self.sess, feed_dict={self.target.x: state.reshape(1, self.state_dims)})[0]

    def value(self, state, action):
        return self.estimator.y.eval(session=self.sess, feed_dict={self.estimator.x: state.reshape(1, self.state_dims)})[0][action]

    def target_value(self, state, action):
        return self.target.y.eval(session=self.sess, feed_dict={self.target.x: state.reshape(1, self.state_dims)})[0][action]

    def update(self, state, action, new_value):
        pass

    def update_batch(self, transitions):
        states = np.ndarray((self.update_batch_size, self.state_dims))
        actions = np.ndarray((self.update_batch_size, 1), dtype=np.int32)
        targets = np.ndarray((self.update_batch_size, 1))
        for i, transition in enumerate(transitions):
            states[i,:] = transition.state
            actions[i,:] = transition.action
            if transition.done:
                targets[i,:] = transition.reward
            else:
                targets[i,:] = transition.reward + self.gamma * np.max(self.target_action_values(transition.next_state))

        self.sess.run(self.estimator.train_op, feed_dict={self.estimator.x: states, self.estimator.a: actions, self.estimator.y_: targets})


if __name__ == "__main__":
    tf.reset_default_graph()
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.Session() as sess:
        deepNn = DeepNNActionValueFunction(sess, 2, 3, 1, 2)
        tf.global_variables_initializer().run()
        deepNn.copy_estimator_to_target()
        print(deepNn.action_values(np.zeros((1, 2))))
        print(deepNn.target_action_values(np.zeros((1, 2))))

        transitions = [Transition(np.zeros((1, 2)), 0, 1, np.zeros((1, 2)), False), Transition(np.zeros((1, 2)), 1, 0, np.zeros((1, 2)), False), Transition(np.zeros((1, 2)), 2, -1, np.zeros((1, 2)), False)]

        print("Starting training...")
        for i in range(100):
            deepNn.update_batch(transitions)
            if i % 10 == 0:
                deepNn.copy_estimator_to_target()
                print(deepNn.action_values(np.zeros((1, 2))))

        print(deepNn.action_values(np.zeros((1, 2))))

