"""
@Author: ChenYao

--------------------
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

train_epochs = 100
batch_size = 100
display_step = 1
learning_rate= 0.0001
drop_prob = 0.5
fch_nodes = 512

def weight_init(shape):
    weights = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights)

def biases_init(shape):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases)

def get_random_batchdata(n_samples, batchsize):
    start_index = np.random.randint(0, n_samples - batchsize)
    return (start_index, start_index + batchsize)
def xavier_init(layer1, layer2, constant = 1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32))

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

w_conv1 = weight_init([7, 7, 1, 16])
b_conv1 = biases_init([16])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_init([7, 7, 16, 32])
b_conv2 = biases_init([32])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_fpool2 = tf.reshape(h_pool2, [-1, 7*7*32])
w_fc1 = xavier_init(7*7*32, fch_nodes)
b_fc1 = biases_init([fch_nodes])
h_fc1 = tf.nn.relu(tf.matmul(h_fpool2, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

w_fc2 = xavier_init(fch_nodes, 10)
b_fc2 = biases_init([10])


y_ = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)
y_out = tf.nn.softmax(y_)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices = [1]))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
n_samples = int(mnist.train.num_examples)
total_batches = int(n_samples / batch_size)

with tf.Session() as sess:
    sess.run(init)
    Cost = []
    Accuracy = []
    for i in range(train_epochs):

        for j in range(10):
            start_index, end_index = get_random_batchdata(n_samples, batch_size)

            batch_x = mnist.train.images[start_index: end_index]
            batch_y = mnist.train.labels[start_index: end_index]
            _, cost, accu = sess.run([ optimizer, cross_entropy,accuracy], feed_dict={x:batch_x, y:batch_y})
            Cost.append(cost)
            Accuracy.append(accu)
        if i % display_step ==0:
            print ('Epoch : %d ,  Cost : %.7f'%(i+1, cost))
    print('training finished')

    fig1,ax1 = plt.subplots(figsize=(10,7))
    plt.plot(Cost)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cost')
    plt.title('Cross Loss')
    plt.grid()
    plt.show()

    fig7,ax7 = plt.subplots(figsize=(10,7))
    plt.plot(Accuracy)
    ax7.set_xlabel('Epochs')
    ax7.set_ylabel('Accuracy Rate')
    plt.title('Train Accuracy Rate')
    plt.grid()
    plt.show()
#----------------------------------visualization------------------------------
    fig2,ax2 = plt.subplots(figsize=(2,2))
    ax2.imshow(np.reshape(mnist.train.images[15], (28, 28)))
    plt.show()

    input_image = mnist.train.images[15:16]
    # input_image = image
    conv1_16 = sess.run(h_conv1, feed_dict={x:input_image})     # [1, 28, 28 ,16]
    conv1_transpose = sess.run(tf.transpose(conv1_16, [3, 0, 1, 2]))
    fig3,ax3 = plt.subplots(nrows=1, ncols=16, figsize = (16,1))
    for i in range(16):
        ax3[i].imshow(conv1_transpose[i][0])
    plt.title('Conv1 16x28x28')
    plt.show()

    pool1_16 = sess.run(h_pool1, feed_dict={x:input_image})     # [1, 14, 14, 16]
    pool1_transpose = sess.run(tf.transpose(pool1_16, [3, 0, 1, 2]))
    fig4,ax4 = plt.subplots(nrows=1, ncols=16, figsize=(16,1))
    for i in range(16):
        ax4[i].imshow(pool1_transpose[i][0])

    plt.title('Pool1 16x14x14')
    plt.show()

    conv2_32 = sess.run(h_conv2, feed_dict={x:input_image})          # [1, 14, 14, 32]
    conv2_transpose = sess.run(tf.transpose(conv2_32, [3, 0, 1, 2]))
    fig5,ax5 = plt.subplots(nrows=1, ncols=32, figsize = (32, 1))
    for i in range(32):
        ax5[i].imshow(conv2_transpose[i][0])
    plt.title('Conv2 32x14x14')
    plt.show()

    pool2_32 = sess.run(h_pool2, feed_dict={x:input_image})         #[1, 7, 7, 32]
    pool2_transpose = sess.run(tf.transpose(pool2_32, [3, 0, 1, 2]))
    fig6,ax6 = plt.subplots(nrows=1, ncols=32, figsize = (32, 1))
    plt.title('Pool2 32x7x7')
    for i in range(32):
        ax6[i].imshow(pool2_transpose[i][0])
    plt.show()
