import tensorflow as tf
import math
import numpy as np

class LeNet:
    def __init__(self, config):
        self._num_classes = config.num_classes # label 개수 (10개-airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        self._l2_reg_lambda = config.l2_reg_lambda #weight decay를 위한 lamda 값

        self.X = tf.compat.v1.placeholder(tf.float32, [None, 32, 32, 3], name="X") # 가로: 32, 세로:32, 채널: RGB
        self.Y = tf.compat.v1.placeholder(tf.float32, [None, self._num_classes], name="Y") # 정답이 들어올 자리, [0 0 0 0 0 0 0 0 0 1] one-hot encoding 형태
        self.keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob") # dropout 살릴 확률
        ##############################################################################################################
        #                         TODO : LeNet5 모델 생성                                                             #
        ##############################################################################################################
        # (32, 32, 3) image

        W1 = tf.Variable(tf.random_normal([5, 5, 3, 6], stddev=math.sqrt(2/75)))
        L1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='VALID')
        L1 = tf.nn.relu(L1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='VALID')

        W2 = tf.Variable(tf.random_normal([3, 3, 6, 16], stddev = math.sqrt(2/54)))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='VALID')
        L2 = tf.nn.relu(L2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # (6, 6, 12) feature map
        # filter3 적용 -> (4, 4, 8) * filter1: 3*3, input_channel: 6, output_channel(# of filters): 16
        # max_pooling 적용 -> (2, 2, 8)
        L2_flat = tf.reshape(L2, [-1, 6 * 6 * 16])

        He = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None,
                                                            dtype=tf.float32)
        FC1 = tf.get_variable("FC1", shape=[6 * 6 * 16, 120], initializer=He)
        FC1 = tf.nn.dropout(FC1, keep_prob=self.keep_prob)
        b1 = tf.Variable(tf.random_normal([120]))
        L3 = tf.nn.relu(tf.matmul(L2_flat, FC1) + b1)
        # (120) features
        # FC2 추가 (120, 84) -> (84)
        FC2 = tf.get_variable("FC2", shape=[120, 84], initializer=He)
        FC2 = tf.nn.dropout(FC2, keep_prob=self.keep_prob)
        b2 = tf.Variable(tf.random_normal([84]))
        L4 = tf.nn.relu(tf.matmul(L3, FC2) + b2)
        # (84) features
        Smax = tf.get_variable("Sm", shape=[84, 10])
        b3 = tf.Variable(tf.random_normal([10]))

        hypothesis = tf.nn.xw_plus_b(L4, Smax, b3, name="hypothesis")  # L3W4 + b4
        # Softmax layer 추가 (84) -> (10)

        with tf.variable_scope('logit'):
          self.predictions = tf.argmax(hypothesis, 1, name="predictions")

        with tf.variable_scope('loss'):
          costs = []
          for var in tf.trainable_variables():
              costs.append(tf.nn.l2_loss(var)) # 모든 가중치들의 l2_loss 누적
          l2_loss = tf.add_n(costs)
          xent = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=self.Y)
          self.loss = tf.reduce_mean(xent, name='xent') + self._l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
