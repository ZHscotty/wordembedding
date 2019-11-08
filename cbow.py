import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import os


class Model(object):
    """docstring for ClassName"""

    def __init__(self, word2index):
        self.EMBED_SIZE = 200
        self.LSTM_HIDDEN_SIZE = 64
        self.VOCAB_SIZE = len(word2index)
        self.LR = 0.001
        self.BATCH_SIZE = 64
        self.should_stop = False
        self.ll = 3
        self.MODEL_DIC = './model/CBOW'
        self.PIC_DIC = './result'
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int64, [None, None])
        self.create_graph()

    def create_graph(self):
        with tf.variable_scope('Dense1'):
            w = tf.get_variable('embedding_martix', [self.VOCAB_SIZE, self.EMBED_SIZE])
            embedd = tf.nn.embedding_lookup(w, self.x)
            self.embedding_matrix = w

            nce_weights = tf.Variable(
                tf.truncated_normal([self.VOCAB_SIZE, self.EMBED_SIZE],
                                    stddev=1.0 / math.sqrt(self.EMBED_SIZE)))
            nce_biases = tf.Variable(tf.zeros([self.VOCAB_SIZE]))

        target = tf.reduce_sum(embedd, axis=1)

        with tf.variable_scope('Dense2'):
            w = tf.get_variable('weight', [self.EMBED_SIZE, self.VOCAB_SIZE])
            b = tf.get_variable('bias', [self.VOCAB_SIZE])
            self.score = tf.einsum('ij,jk->ik', target, w) + b

        # 准确率
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.score, axis=1), tf.argmax(self.y, axis=1)), tf.float32))
        # 交叉熵计算损失
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=score)
        # loss = tf.reduce_mean(loss)
        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, inputs=target, labels=self.y, num_sampled=64,
                           num_classes=self.VOCAB_SIZE))
        self.train_op = tf.train.AdamOptimizer(self.LR).minimize(self.loss)

        # return score, acc, loss, train_step, embedding_matrix

    def train(self, x_train, y_train, x_dev, y_dev, epoch):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            acc_train = []
            acc_dev = []
            loss_train = []
            loss_dev = []

            train_step = 0
            earlystop = 0
            loss_min = 99999
            n = 0

            while train_step < epoch and self.should_stop is False:
                print('Epoch:{}'.format(train_step))
                index = 0
                acc_total = 0
                loss_total = 0
                for x_batch, y_batch in self.get_batch(x_train, y_train, self.BATCH_SIZE):
                    _, acc_t, loss_t = sess.run([self.train_op, self.acc, self.loss],
                                                {self.x: x_batch, self.y: y_batch})
                    print('step:{}  [{}/{}] --acc:{:.5f} --loss:{:.5f}'.format(index, index*self.BATCH_SIZE,
                                                                               len(x_train), acc_t, loss_t))
                    index += 1
                    acc_total += acc_t
                    loss_t += loss_t

                acc_t = acc_total/index
                loss_t = loss_total/index
                acc_train.append(acc_t)
                loss_train.append(loss_t)

                acc_total = 0
                loss_total = 0
                index = 0
                for x_batch, y_batch in self.get_batch(x_dev, y_dev, 10000):
                    acc_d, loss_d = sess.run([self.acc, self.loss], {self.x: x_batch, self.y: y_batch})
                    acc_total += acc_d
                    loss_total += loss_d
                    index += 1

                acc_d = acc_total/index
                loss_d = loss_total/index

                acc_dev.append(acc_d)
                loss_dev.append(loss_d)

                print('Epoch{}----acc:{:.5f},loss:{:.5f},val-acc:{:.5f},val_loss:{:.5f}'.format(train_step, acc_t,
                                                                                                loss_t, acc_d, loss_d))
                if loss_d > loss_stop:
                    if n > self.ll:
                        self.should_stop = True
                        es_step = train_step
                    else:
                        n += 1
                else:
                    if not os.path.exists(self.MODEL_DIC):
                        os.makedirs(self.MODEL_DIC)
                    saver.save(sess, os.path.join(self.MODEL_DIC, 'cbow_model'))
                    n = 0
                    loss_stop = loss_d
                train_step += 1

            if self.should_stop is True:
                print('Early Stop at Epoch{}'.format(es_step))

        if not os.path.exists(self.PIC_DIC):
            os.makedirs(self.PIC_DIC)

        plt.plot(acc_train)
        plt.plot(acc_dev)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.PIC_DIC, 'acc.svg'))
        plt.close()

        plt.plot(loss_train)
        plt.plot(loss_dev)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(os.path.join(self.PIC_DIC, 'loss.svg'))
        plt.close()

    def predict(self):
        return None

    def print_shape(self, name, tensor):
        print('{} shape={}'.format(name, tensor.shape))

    def get_batch(self, data, label, batch_size):
        begin = 0
        iter_num = len(data)//batch_size
        if len(data)%batch_size != 0:
            iter_num += 1

        for i in range(iter_num):
            end = begin + batch_size
            if end > len(data):
                end = len(data)
            x_batch = data[begin:end]
            y_batch = label[begin:end]
            begin = end
            yield x_batch, y_batch



