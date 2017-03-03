"""
WIP

The idea here is to write something which combines a policy and a query function,
handling everything which is necessary for Active RL, so no changes to the env are needed.

So the active policy should:
    1. make both action and query decisions
    2. determing the correct "rewards" to use


"""

import numpy as np

from model import *#conv2d

def tf_log2(x):
    return tf.log(x) / np.log(2)

# TODO2: temp
def multinomial_entropy(pvec, temp=1):
    #pvec *= temp 
    pvec /= tf.reduce_sum(tf.abs(pvec))
    return -tf.reduce_sum(tf_log2(pvec) * pvec)

class DefaultRewardNet(object):
    def __init__(self, input_var, ob_space, num_hids=256,
            **kwargs):
        self.x = input_var

        rank = len(ob_space)

        if rank == 3: # pixel input
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        elif rank == 1: # plain features
            #x = tf.nn.elu(linear(x, 256, "l1", normalized_columns_initializer(0.01)))
            pass
        else:
            raise TypeError("observation space must have rank 1 or 3, got %d" % rank)

        self.x = linear(x, num_hids, "reward_net_linear_layer", normalized_columns_initializer(0.01))
        # TODO2: deal with reward-space != {-1,0,1} (and/or enforce this with ternarize_wrapper)
        self.logits = linear(x, 3, "reward_logits", normalized_columns_initializer(0.01))
        self.reward_prediction = tf.argmax(self.logits, axis=1) - 1
        self.reward_distribution = tf.nn.softmax(self.logits)


class ActiveLSTMPolicy(object):
    """
    For now, we just modify the original policy, adding a reward predictor (reward_net), 
    and specify the query_fn as one of a number of options
    """
    def __init__(self, ob_space, ac_space, lstm_size=256,
            reward_net=None, query_fn='always',
            **kwargs):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space))
        # TODO: how to update this?
        self.num_queries = tf.Variable(0)
        # TODO: make a dataset for off-line reward function learning
        if reward_net is not None:
            assert False
            self.reward_net = reward_net
        else:
            self.reward_net = DefaultRewardNet(input_var=x, ob_space=ob_space)
        self.reward_prediction = self.reward_net.reward_prediction
        self.reward_distribution = self.reward_net.reward_distribution

        #self.reward
        if query_fn == 'always':
            self.query = tf.Variable(1)
        if query_fn.startswith('constant_probability='):
            prob = float(query_fn.split('constant_probability=')[1])
            self.query = tf.random_uniform([1]) < prob
        if query_fn.startswith('firstN='):
            self.N = int(query_fn.split('firstN=')[1])
            self.query = self.num_queries < self.N
        if query_fn.startswith('entropy_based_temp='):
            # for now, we hard-wire the entropy to be based on 3 possible reward values
            max_ent = 1.5849625007211561
            temp = float(query_fn.split('entropy_based_temp=')[1])
            prob = multinomial_entropy(self.reward_distribution, temp=temp) / max_ent
            self.query = tf.random_uniform([1]) < prob


        rank = len(ob_space)

        if rank == 3: # pixel input
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        elif rank == 1: # plain features
            #x = tf.nn.elu(linear(x, 256, "l1", normalized_columns_initializer(0.01)))
            pass
        else:
            raise TypeError("observation space must have rank 1 or 3, got %d" % rank)

        # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
        x = tf.expand_dims(flatten(x), [0])

        size = lstm_size
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.reward_prediction, self.reward_distribution, self.query, self.num_queries, self.sample, self.vf] + self.state_out ,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]

    """
    def tnq():
        sess = tf.get_default_session()
        return sess.run([self.num_queries], )
    """


active_policies = dict(
    active_lstm_ = ActiveLSTMPolicy()
    #active_lstm_firstN = ActiveLSTMPolicy()
        )




# TODO2
class ActivePolicy(object):
    def __init__(self, policy_net, query_net):
        self.__dict__.update(locals())

    def get_initial_features(self):
        return self.policy_net.get_initial_features()

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.policy_net.sample, self.policy_net.vf] + self.policy_net.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]
