import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets('MNIST_DATA/', one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), 'W')
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    array_to_image(v_xs[2])
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    print(y_pre[2])
    print(v_ys[2])
    print(np.argmax(y_pre[2]))
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def array_to_image(array):
    narray = array.copy().reshape([28,28]) * 255
    img = Image.fromarray(narray)
    img = img.convert('L')
    img.show()
    # plt.imshow(array)

xs = tf.placeholder(tf.float32, [None, 784]) # 28*28 不限定多少个example，限定每个example大小为784
ys = tf.placeholder(tf.float32, [None, 10])

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if step == 999:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
