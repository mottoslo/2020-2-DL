import tensorflow as tf
import numpy as np

class CGAN(object):

    def __init__(self, config):
        self.D_hidden_dim = config["D_hidden_dim"] # discriminator의 hidden size
        self.G_hidden_dim = config["G_hidden_dim"] # generator의 hidden size
        self.Z_dim = config["Z_dim"] # noise vector size
        self.Y_dim = config["Y_dim"] # output layer size

        # Placeholders for input
        self.input_x = tf.placeholder(tf.float32, shape=[None, 784])
        self.input_y = tf.placeholder(tf.float32, shape=[None, self.Y_dim])
        self.input_z = tf.placeholder(tf.float32, shape=[None, self.Z_dim])

        # discriminator의 weight matrix들을 초기화
        self.D_W1 = tf.get_variable("D_W1", shape=[784 + self.Y_dim, self.D_hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.D_b1 = tf.Variable(tf.zeros(shape=[self.D_hidden_dim]))

        self.D_W2 = tf.get_variable("D_W2", shape=[self.D_hidden_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.D_b2 = tf.Variable(tf.zeros(shape=[1]))

        # discriminator의 파라미터 집합 (generator의 파라미터는 freeze 하기 위함)
        self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]

        # generator의 weight matrix들을 초기화
        self.G_W1 = tf.get_variable("G_W1", shape=[self.Z_dim + self.Y_dim, self.G_hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
        self.G_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_dim]))

        self.G_W2 = tf.get_variable("G_W2", shape=[self.G_hidden_dim, 784], initializer=tf.contrib.layers.xavier_initializer())
        self.G_b2 = tf.Variable(tf.zeros(shape=[784]))

        # generator의 파라미터 집합 (discriminator 의 파라미터는 freeze 하기 위함)
        self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

        self.G_sample = self.generator(self.input_z, self.input_y) # generator가 생성한 image
        D_real, D_logit_real = self.discriminator(self.input_x, self.input_y) # discriminator의 실제이미지에 대한 예측
        D_fake, D_logit_fake = self.discriminator(self.G_sample, self.input_y) # discriminator의 가짜이미지에 대한 예측

        # discriminator의 실제이미지 예측에 대한 loss
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        # discriminator의 가짜이미지 예측에 대한 loss
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        # discriminator의 loss
        self.D_loss = D_loss_real + D_loss_fake
        # generator의 loss
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n]) # noise 생성

    def generator(self, z, y): # generator에서 affine 연산 및 activation
        inputs = tf.concat(axis=1, values=[z, y])
        G_h1 = tf.nn.relu(tf.matmul(inputs, self.G_W1) + self.G_b1)
        G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        return G_prob

    def discriminator(self, x, y): # discriminator에서 affine 연산 및 activation
        inputs = tf.concat(axis=1, values=[x,y])
        D_h1 = tf.nn.relu(tf.matmul(inputs, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit




