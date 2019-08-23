import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#training dataset
xs = [[0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
ys = [[0.5,0.9,1.3,0.4,0.8,2.6,0.7,0.1,3.9,0.6]]

#place holders for further execution
X = tf.compat.v1.placeholder(tf.float32,name='X')
Y = tf.compat.v1.placeholder(tf.float32,name='Y')

Y_pred = tf.Variable(tf.random.normal([1]), name='bias')
for pow_i in range(1, 5):
    W = tf.Variable(tf.random.normal([1]), name='weight_%d' % pow_i)
    Y_pred = tf.add(tf.multiply(tf.pow(X,pow_i), W), Y_pred)

cost_f = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (10 - 1)
cost_f = tf.add(cost_f, tf.multiply(1e-6, tf.linalg.global_norm([W])))

learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_f)

saver = tf.train.Saver()
n_epochs = 1000
with tf.compat.v1.Session() as sess:   
    sess.run(tf.global_variables_initializer())
    #shape of our graph
    tf.io.write_graph(graph_or_graph_def=sess.graph_def,
                     logdir='.',
                     name='polynomial_regression.pb',
                     as_text=False)
    prev_training_cost = 0.0
    for epoch_i in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        training_cost = sess.run(
            cost_f, feed_dict={X: xs, Y: ys})
        print(training_cost)
        
        if np.abs(prev_training_cost - training_cost) < 0.000001:
            #saving after training
            saver.save(sess=sess, save_path='./polynomial_regression.ckpt')
            break
        prev_training_cost = training_cost
