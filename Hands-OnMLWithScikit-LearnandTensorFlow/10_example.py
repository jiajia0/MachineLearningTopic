# @Time    : 2019/6/18 17:06
# @Author  : Leafage
# @File    : 10_example.py
# @Software: PyCharm
# @Describe: 第十章中的例子
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 定义输入、输出、隐藏节点的信息
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# 使用占位符：placeholder 来定义输入和输出的数据
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


# 每次创建一个层
def neuron_layer(X, n_neurons, name, activation=None):
    """
    :param X: 输入的数据
    :param n_neurons: 神经元数量
    :param name: 层次的名称
    :param activation: 激活函数
    :return:
    """
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])

        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b  # 矩阵相乘
        if activation == 'relu':
            return tf.nn.relu(z)
        else:
            return z


with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
    hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
    logits = neuron_layer(hidden2, n_outputs, "outputs")


with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")


learning_rate = 0.01
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()


mnist = input_data.read_data_sets("Hands-OnMLWithScikit-LearnandTensorFlow/datasets/mnist")

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print(epoch, "Train accuracy:", str(acc_train), "Test accuracy:", acc_test)
    save_path = saver.save(sess, "my_model_final.ckpt")


# # 预测数据
# with tf.Session() as sess:
#     saver.restore(sess, "./my_model_final.ckpt")
#     X_new_scaled = []  # 新数据
#     Z = logits.eval(feed_dict={X: X_new_scaled})
#     y_pred = np.argmax(Z, axis=1)