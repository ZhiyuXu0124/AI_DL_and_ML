import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 2], name="input_x")
d = tf.placeholder(tf.float32, [None, 2], name="input_y")
# 对于sigmoid激活函数而言，效果可能并不理想
net = tf.layers.dense(x, 4, activation=tf.nn.relu)
net = tf.layers.dense(net, 4, activation=tf.nn.relu)
y = tf.layers.dense(net, 2, activation=None)
loss = tf.reduce_mean(tf.square(y-d))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(d, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.1)
gradient = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
train_step = optimizer.apply_gradients(gradient)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


import numpy as np
files = np.load("homework.npz")
X = files['X']
label = files['d']
label_one_hot = []
for x1, x2 in X:
    if x1 > 0 and x2 > 0:
        label_one_hot.append([1, 0])
    elif x1 < 0 and x2 < 0:
        label_one_hot.append([1, 0])
    else:
        label_one_hot.append([0, 1])
label_one_hot = np.array(label_one_hot)
for itr in range(500):
    idx = np.random.randint(0, 2000, 20)
    inx = X[idx]
    ind = label_one_hot[idx]
    sess.run(train_step, feed_dict={x:inx, d:ind})
    if itr%10 == 0:
        acc = sess.run(accuracy, feed_dict={x:X, d:label_one_hot})
        print("step:{}  accuarcy:{}".format(itr, acc))