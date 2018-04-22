import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(layer_name, inputs, in_size, out_size, activation_function=None):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), 'W')
            tf.summary.histogram('/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram('/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram('/outputs', outputs)
    return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis];
# print(np.linspace(-1, 1, 10))
# print(np.linspace(-1, 1, 10)[:,np.newaxis])
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
# plt.plot(x_data, y_data , linestyle='', marker='.')
plt.ion()       # 不暂停
plt.show()
with tf.name_scope('inputs'):
    xs = tf.placeholder(np.float32, [None, 1], 'x_input')
    ys = tf.placeholder(np.float32, [None, 1], 'y_input')

l1 = add_layer('layer1', xs, 1, 20, activation_function=tf.nn.relu)
prediction = add_layer('layer2', l1, 20, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.MomentumOptimizer(0.05,0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logdir="log/", graph=sess.graph)
feed = {xs: x_data, ys: y_data}
for step in range(3000):
    sess.run(train_step, feed_dict=feed)
    print(step, sess.run(loss, feed_dict=feed))
    if step % 20 == 0:
        rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs, step)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_val = sess.run(prediction, feed_dict=feed)
        lines = ax.plot(x_data, prediction_val, 'r-', lw=5)
        plt.pause(0.1)

plt.show(block=True)