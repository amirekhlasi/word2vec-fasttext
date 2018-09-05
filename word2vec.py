# coding: utf-8

import tensorflow as tf
import numpy as np
import pickle

class skipgram_negative_sampling(object):
    def __init__(self, session,  vocab_size, dim, batch_size = 4096,
                 negative_sampled_size = 64, device = "cpu:0", construct_model = True):
        self.session = session
        self.vocab_size = vocab_size
        self.dim = dim
        self.batch_size = batch_size
        self.negative_sampled_size = negative_sampled_size
        self.device = device
        # batch_genrator variables
        self.batch_num = 0
        #init values for W1 and W2
        self.init_W1 = tf.random_normal(shape = [self.vocab_size, self.dim])
        self.init_W2 = tf.random_normal(shape = [self.vocab_size, self.dim])
        # Model
        if construct_model:
            self.construct_model()

    def set_weights(W1,W2):
        self.init_W1 = W1
        self.init_W2 = W2

    def construct_model(self):
        with tf.device(self.device):
            self.input = tf.placeholder(tf.int32, [None])
            self.label = tf.placeholder(tf.int32, [None, 1])
            sefl.neg_sampled_size = tf.placeholder(tf.int32, shape = ())
            self.W1 = tf.Variable(self.init_W1)
            self.W2 = tf.Variable(self.init_W2)
            self.z = tf.nn.embedding_lookup(self.W1, self.input)
            self.bias = tf.zeros([self.vocab_size])
            self.loss = tf.reduce_mean(
                tf.nn.nce_loss(weights = self.W2, biases = self.bias, inputs = self.z,
                    labels = self.label, num_sampled = self.neg_sampled_size, num_classes=self.vocab_size))


    def initialize(self):
        self.session.run(tf.initialize_variables([self.W1, self.W2]))

    def reset_batch(self):
        self.batch_num = 0
        self.perm = np.random.permutation(self.train_size - 1)
        self.train_input = self.train_input[self.perm]
        self.train_label = self.train_label[self.perm]

    def get_batch(self):
        start = self.batch_size*self.batch_num
        end = (self.batch_num + 1)*self.batch_size
        self.batch_num = self.batch_num + 1
        return self.train_input[start:end], self.train_label[start:end]

    def end_batch(self):
        return (self.batch_num + 1)*self.batch_size > self.train_size

    def set_train(self, train):
        train = np.array(train, dtype = 'int32')
        self.train_input = train[:,0]
        self.train_label = train[:,1]
        self.train_size,_ = train.shape

    def train(self, epoch_num, learning_rate, batch_size = self.batch_size, negative_sample_size =
                self.negative_sampled_size ,print_epoch = False):
        old_batch_size = self.batch_size
        self.batch_size = batch_size
        with tf.device(self.device):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(self.loss)
        for epoch in range(epoch_num):
            self.reset_batch()
            gen_loss = 0
            while not self.end_batch():
                input, label = self.get_batch()
                feed_dict = {self.input: input, self.label: label.reshape(self.batch_size,1),
                            self.neg_sampled_size = negative_sample_size}
                loss, _ = self.session.run([self.loss, optimizer], feed_dict = feed_dict)
                gen_loss = gen_loss + loss
            if print_epoch:
                print("epoch:  ", epoch + 1)
                print("loss:   ", gen_loss/self.batch_num)
        self.batch_size = old_batch_size

    def export_embedding(self):
        W1, W2 = self.session.run([self.W1, self.W2])
        return np.array(W1), np.array(W2)
