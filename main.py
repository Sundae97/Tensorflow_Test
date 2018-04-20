import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# a = np.linspace(0,10)
# b = np.linspace(0,5)
# plt.figure()
# plt.plot(a,b)
# plt.show()

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

plt.figure()
plt.plot(x_data , y_data, color='red', marker='x',linestyle='')

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    print(step, sess.run(Weights), sess.run(biases) , sess.run(loss))

print("\n----------------END----------------")

w = sess.run(Weights)
b = sess.run(biases)
# print("w : " + str(w) + "   b : " + str(b))
#
lin_x = np.linspace(0,1,100)
result_y = w * lin_x + b

plt.plot(lin_x , result_y ,color='green',linestyle='--')
plt.show()