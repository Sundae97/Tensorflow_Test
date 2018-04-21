import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
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

xs = tf.placeholder(np.float32, [None, 1])
ys = tf.placeholder(np.float32, [None, 1])

l1 = add_layer(xs, 1, 20, activation_function=tf.nn.relu6)
prediction = add_layer(l1, 20, 1, activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

feed = {xs: x_data, ys: y_data}
for step in range(2000):
    sess.run(train_step, feed_dict=feed)
    print(step, sess.run(loss, feed_dict=feed))
    if step % 20 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_val = sess.run(prediction, feed_dict=feed)
        lines = ax.plot(x_data, prediction_val, 'r-', lw=5)
        plt.pause(0.1)

plt.show(block=True)