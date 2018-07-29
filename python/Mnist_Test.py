# encoding:utf-8
# author:zee
import tensorflow as tf
# import keras
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data
# from keras.datasets import mnist
# from keras.utils import np_utils
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(-1, 1, 28, 28)/255.
# X_test = X_test.reshape(-1, 1, 28, 28)/255.
# y_train = np_utils.to_categorical(y_train, num_classes=10)
# y_test = np_utils.to_categorical(y_test, num_classes=10)
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
# 启动图计算
sess = tf.InteractiveSession()
# 占位符，None表示其值（图片个数）大小不定，784表示是一张MNIST展开图片的维度，10表示one-hot向量，代表对应MNIST图片的类别
x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])
# 定义初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 定义偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 定义卷基层
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# 定义池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 第一层卷积神网
w_cov1 = weight_variable([5, 5, 1, 32])
b_cov1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_cov1 = tf.nn.relu(conv2d(x_image, w_cov1)+b_cov1)
h_pool1 = max_pool_2x2(h_cov1)
# 第二层卷积神网
w_cov2 = weight_variable([5, 5, 32, 64])
b_cov2 = bias_variable([64])
h_cov2 = tf.nn.relu(conv2d(h_pool1, w_cov2)+b_cov2)
h_pool2 = max_pool_2x2(h_cov2)
# 密集全连接层
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)
# Dropout处理
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 输出层
w_output = weight_variable([1024, 10])
b_output = bias_variable([10])
y_cov = tf.nn.softmax(tf.matmul(h_fc1_drop, w_output)+b_output)
# 计算交叉啇
cross_entropy = -tf.reduce_sum(y_*tf.log(y_cov))
# 学习优化算法:以1e-4的学习速率最小化交叉啇
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 计算正确率
correct_prediction = tf.equal(tf.argmax(y_cov, 1), tf.argmax(y_, 1))
Accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
# 初始化我们创建的变量
sess.run(tf.global_variables_initializer())
# 开始训练模型
for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_:  batch[1], keep_prob: 0.5})
    if i % 50 == 0:
        train_accuracy = Accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d,training accuracy %gg' % (i, train_accuracy))
#    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print('test accuracy %g' % Accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))







