from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from math import pi

import sys
from ops import svgd_gradient,wgf_gradient, rbf_kernel


class Layer():

    def __init__(self, n_p, n_in, n_out, activation_fn=tf.nn.relu, name='l1'):
        # n_p: number of particles
        # n_in: input dimension
        # n_out: output dimension

        #self.__dict__.update(locals())
        self.n_p, self.n_in, self.n_out = n_p, n_in, n_out
        self.activation_fn = activation_fn
        with tf.variable_scope(name) as scope:
            w0 = (1.0 / np.sqrt(self.n_in + 1) * np.random.randn(self.n_p, self.n_in, self.n_out)).astype('float32')
            self.w = tf.get_variable('w',
                    shape=(self.n_p, self.n_in, self.n_out), dtype=tf.float32,
                    initializer = tf.glorot_uniform_initializer() )
                    #initializer=w0, dtype=tf.float32)

            self.params = [self.w]


    def forward(self, inputs):
        assert tf.keras.backend.ndim(inputs) == 3
        # inputs: n_p x B x n_in
        # w: n_p x n_in x n_out
        a = tf.matmul(inputs, self.w)
        if self.activation_fn is not None:
            h = self.activation_fn(a)
        else:
            h = a
        return a, h


class WGF():

    def __init__(self, config):
        self.config = config

        # create placeholders for the input
        self.X = tf.placeholder(
            name='X', dtype=tf.float32,
            shape=[None, self.config.dim],
        )

        self.y = tf.placeholder(
            name='y', dtype=tf.float32,
            shape=[None],
        )

        #self.log_v_noise = tf.get_variable('log_v_noise', 
        #        initializer=tf.constant(np.log(1.0,).astype('float32')),
        #        dtype=tf.float32)

        #self.v_noise_vars = [self.log_v_noise]

        self.step = tf.placeholder_with_default(1., shape=(), name='step')
        #self.neg_log_var = tf.placeholder_with_default(0., shape=(), name='neg_log_var')
        self.n_neurons = [self.config.dim, self.config.n_hidden, self.config.n_hidden, 1]
        #self.n_neurons = [self.config.dim, self.config.n_hidden, 1]

        # build network
        self.nnet = []
        for i in range(len(self.n_neurons) - 1):
            activation_fn = tf.nn.relu
            if i == len(self.n_neurons) - 2:
                activation_fn = None
            self.nnet.append( Layer(self.config.n_particles, self.n_neurons[i], self.n_neurons[i+1], activation_fn=activation_fn, name='l_%d' % i) )

        # forward, A, H
        n_layers = len(self.nnet)
        cache = []
        self.train_vars = []
        h = tf.tile(tf.expand_dims(self.X, 0), (self.config.n_particles, 1, 1))
        cache.append(h)
        for i in range(n_layers):
            a, h = self.nnet[i].forward(h)
            cache.append(a)
            if i != n_layers-1: cache.append(h) # last layer
            self.train_vars += self.nnet[i].params
        self.y_pred = tf.squeeze(h) # n_p x B

        #self.log_prob = tf.reduce_sum(self.get_log_liklihood(self.y, self.y_pred))
        self.log_lik, self.log_prior = self.get_log_liklihood(self.y, self.y_pred)
        self.log_prob =  self.log_lik + self.log_prior  # []
        self.net_grads = tf.gradients(self.log_prob, self.train_vars)

        #############################################################
        # vanilla svgd 
        self.wgf_grads = []
        for p, g in zip(self.train_vars, self.net_grads):
            wgf_grad = svgd_gradient(p, g, kernel=self.config.kernel)
            self.wgf_grads.append( -wgf_grad ) # maximize

        

        tf.summary.scalar("log_prob", tf.reduce_sum(self.log_prob))


    def get_log_liklihood(self, y_true, y_pred):
        #v_noise = tf.exp(self.log_v_noise)
        log_v_noise, v_noise = np.log(0.5), 0.5
        # location = 0, scale = 1
        log_lik_data = -self.config.n_train * 0.5 * tf.log(2.*np.pi) * log_v_noise \
                       -self.config.n_train * 0.5 * tf.reduce_mean((y_pred - tf.expand_dims(y_true, 0))**2 / v_noise, axis=1)

        log_prior_w = 0
        for p in self.train_vars: 
            log_prior_w += ( -0.5*tf.reduce_sum(tf.reshape(p**2, (self.config.n_particles, -1)), axis=1) )

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        #log_posterior = log_lik_data + log_prior_w
        #return log_posterior
        return log_lik_data, log_prior_w


    def get_feed_dict(self, batch_chunk, step=None):
        fd = {
            self.X: batch_chunk['X'],  
            self.y: batch_chunk['y'],  
        }
        if step is not None:
            fd[self.step] = step
        return fd


