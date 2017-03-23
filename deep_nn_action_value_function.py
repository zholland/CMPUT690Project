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
    def __init__(self, scope="estimator"):
        self.scope = scope
        # self.session = tf.InteractiveSession()
        with tf.variable_scope(scope):
            self._create_model()

    def _create_model(self):
        # Input
        self.x = tf.placeholder(tf.float32, [None, 2])
        self.a = tf.placeholder(tf.int32, [None, 1])

        # Fully connected
        W_1 = tf.Variable(tf.truncated_normal(shape=[2, 4], stddev=0.1))
        b_1 = tf.Variable(tf.constant(0.1, shape=[4]))
        h_1 = tf.nn.relu(tf.matmul(self.x, W_1) + b_1)

        # Fully connected
        W_2 = tf.Variable(tf.truncated_normal(shape=[4, 4], stddev=0.1))
        b_2 = tf.Variable(tf.constant(0.1, shape=[4]))
        h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

        # Fully connected linear output
        W_3 = tf.Variable(tf.truncated_normal(shape=[4, 3], stddev=0.1))
        b_3 = tf.Variable(tf.constant(0.1, shape=[3]))
        self.y = tf.matmul(h_2, W_3) + b_3

        # Get the predictions for the chosen actions only
        # gather_indices = tf.range(1) * tf.shape(self.y)[1] + self.a
        self.a_value_estimates = self.y[0, 1]

        # Target placeholder
        self.y_ = tf.placeholder(tf.float32, [None, 1])

        # Loss and optimizer
        loss = tf.reduce_mean(tf.squared_difference(self.y_, self.a_value_estimates))
        self.train_op = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss, global_step=tf.contrib.framework.get_global_step())


class DeepNNActionValueFunction(AbstractActionValueFunction):
    def __init__(self, sess, gamma):
        self.sess = sess
        self.gamma = gamma
        self.estimator = QEstimator(scope="estimator")
        self.target = QEstimator(scope="target")

    def copy_estimator_to_target(self):
        estimator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="estimator")
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target")

        assign_ops = []
        for estimator_v, target_v in zip(estimator_vars, target_vars):
            op = target_v.assign(estimator_v)
            assign_ops.append(op)

        self.sess.run(assign_ops)

    def action_values(self, state):
        return self.estimator.y.eval(feed_dict={self.estimator.x: state})[0]

    def target_action_values(self, state):
        return self.target.y.eval(feed_dict={self.target.x: state})[0]

    def value(self, state, action):
        return self.estimator.y.eval(feed_dict={self.estimator.x: state})[0][action]

    def target_value(self, state, action):
        return self.target.y.eval(feed_dict={self.target.x: state})[0][action]

    def update(self, state, action, new_value):
        pass

    def update_batch(self, transitions, batch_size):
        states = np.ndarray((batch_size, 2))
        actions = np.ndarray((batch_size, 1), dtype=np.int32)
        targets = np.ndarray((batch_size, 1))
        for i, transition in enumerate(transitions):
            states[i,:] = transition.state
            if transition.done:
                targets[i,:] = transition.reward
            else:
                targets[i,:] = transition.reward + self.gamma * np.max(self.target_action_values(transition.next_state))

        self.sess.run(self.estimator.train_op, feed_dict={self.estimator.x: states, self.estimator.a: actions, self.estimator.y_: targets})


tf.reset_default_graph()
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
    deepNn = DeepNNActionValueFunction(sess, 1)
    tf.global_variables_initializer().run()
    deepNn.copy_estimator_to_target()
    print(deepNn.action_values(np.zeros((1, 2))))
    print(deepNn.target_action_values(np.zeros((1, 2))))

    transitions = [Transition(np.zeros((1, 2)), 0, 100, np.zeros((1, 2)), False)]
    transitions2 = [Transition(np.zeros((1, 2)), 1, -100, np.zeros((1, 2)), False)]

    for _ in range(2):
        deepNn.update_batch(transitions, 1)
        print(deepNn.action_values(np.zeros((1, 2))))
        deepNn.update_batch(transitions2, 1)
        print(deepNn.action_values(np.zeros((1, 2))))

    print(deepNn.target_action_values(np.zeros((1, 2))))
    deepNn.copy_estimator_to_target()
    print(deepNn.action_values(np.zeros((1, 2))))
    print(deepNn.target_action_values(np.zeros((1, 2))))

    for _ in range(2):
        deepNn.update_batch(transitions, 1)
        print(deepNn.action_values(np.zeros((1, 2))))
        deepNn.update_batch(transitions2, 1)
        print(deepNn.action_values(np.zeros((1, 2))))

    print(deepNn.target_action_values(np.zeros((1, 2))))
    deepNn.copy_estimator_to_target()
    print(deepNn.action_values(np.zeros((1, 2))))
    print(deepNn.target_action_values(np.zeros((1, 2))))

