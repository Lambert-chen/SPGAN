"""
@Time:2019.5.6
@Author: ChenYao
--------------------
Reference model in GandD
The main program includes pruning and retraining

"""
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import scipy.misc
import pandas as pd
import warnings
from SPGAN.GandD import get_generator,get_discriminator
from SPGAN.growing import generator,discriminator,prune_feature_maps,feature_maps
from SPGAN.load_data import load_cfar10_batch
from SPGAN.GetListMinIndex import getListMaxNumIndex
from SPGAN.growing import generator_variables_dict1,discriminator_variables_dict1
warnings.filterwarnings("ignore")
print("TensorFlow Version: {}".format(tf.__version__))

cifar10_path = 'cifar-10-python\cifar-10-batches-py'
x_train, y_train = load_cfar10_batch(cifar10_path, 1)
for i in range(2, 6):
    features, labels = load_cfar10_batch(cifar10_path, i)
    x_train, y_train = np.concatenate([x_train, features]), np.concatenate([y_train, labels])
fig, axes = plt.subplots(nrows=3, ncols=20, sharex=True, sharey=True, figsize=(80,12))
imgs = x_train[:60]

for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60]], axes):
    for img, ax in zip(image, row):
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
fig.tight_layout(pad=0.1)

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3)


x_train = minmax.fit_transform(x_train_rows)

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)

images = x_train[y_train==1]

def plot_images(samples,steps):
    samples = (samples + 1) / 2
    fig, axes = plt.subplots(nrows=1, ncols=15, sharex=True, sharey=True, figsize=(30,2))
    for img, ax in zip(samples, axes):
        ax.imshow(img.reshape((32, 32, 3)), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout(pad=0)
    scipy.misc.imsave('out\cifar' + str(steps) +  '.jpg', img)
    # plt.show()
def show_generator_output(sess, n_images, inputs_noise, output_dim):
    cmap = 'Greys_r'
    noise_shape = inputs_noise.get_shape().as_list()[-1]
    examples_noise = np.random.uniform(-1, 1, size=[n_images, noise_shape])

    samples = sess.run(get_generator(inputs_noise, output_dim, False),
                       feed_dict={inputs_noise: examples_noise})
    return samples

batch_size = 64
noise_size = 100
epochs = 50
n_samples = 64
learning_rate = 0.001
beta1 = 0.4
data_shape = [64,32,32,3]
losses = []
steps = 0

inputs_real = tf.placeholder(tf.float32, [None, data_shape[1], data_shape[2], data_shape[3]], name='inputs_real')
inputs_noise = tf.placeholder(tf.float32, [None, noise_size], name='inputs_noise')
# g_loss, d_loss = get_loss(inputs_real, inputs_noise, data_shape[-1])

g_outputs = get_generator(inputs_noise, data_shape[-1], is_train=True)
d_logits_real, d_outputs_real = get_discriminator(inputs_real)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_outputs_fake) * (1 - 0.1)))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real) * (
                                                                                 1 - 0.1)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
d_loss = tf.add(d_loss_real, d_loss_fake)
# g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)

train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# Optimizer
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(10):
        for batch_i in range(images.shape[0] // batch_size - 1):
            steps += 1
            batch_images = images[batch_i * batch_size: (batch_i + 1) * batch_size]

            # scale to -1, 1
            batch_images = batch_images * 2 - 1

            # noise
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # run optimizer
            _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
            _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})

            if steps % 10 == 0:
                train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                losses.append((train_loss_d, train_loss_g))
                dataframe = pd.DataFrame({'D_loss': train_loss_d, 'G_loss': train_loss_g},index = [0])
                dataframe.to_csv(r"loss.csv",sep=',',mode='a+',header=False)
                samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
                plot_images(samples,steps)
                print("Epoch {}/{}".format(e + 1, 10),
                          "Discriminator Loss: {:.4f}".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
#-------------------------------------------------------growing retrain----------------------
inputs_real = tf.placeholder(tf.float32, [64,32,32,3], name='inputs_real')
inputs_noise = tf.placeholder(tf.float32, [64, 100], name='inputs_noise')

g_outputs = generator(inputs_noise, data_shape[-1], is_train=True)
d_logits_real, d_outputs_real = discriminator(inputs_real)
d_logits_fake, d_outputs_fake = discriminator(g_outputs, reuse=True)
# 计算Loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_outputs_fake) * (1 - 0.1)))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real) * (
                                                                                 1 - 0.1)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
d_loss = tf.add(d_loss_real, d_loss_fake)
train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# Optimizer
g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 迭代epoch
    for e in range(30):
        for batch_i in range(images.shape[0] // batch_size - 1):
            steps += 1
            batch_images = images[batch_i * batch_size: (batch_i + 1) * batch_size]
            # scale to -1, 1
            batch_images = batch_images * 2 - 1
            # noise
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
            # run optimizer
            _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
            _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
            if steps % 10 == 0:
                train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                losses.append((train_loss_d, train_loss_g))
                # 显示图片
                samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
                plot_images(samples,steps)
                print("Epoch {}/{}".format(e + 1, 40),
                          "Discriminator Loss: {:.4f}".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
#------------------------------------------------------
#layer2[64, 8, 8, 256]---->[64, 8, 8, 231]
#layer2[64, 16, 16, 128]---->[64, 16, 16, 116]
input_image = x_train[59:60]
batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
g2,g3 ,d2,d3 = feature_maps()
gf2 = sess.run(g2, feed_dict={inputs_real: input_image,inputs_noise: batch_noise})
gconv2_transpose = sess.run(tf.transpose(gf2, [3, 0, 1, 2]))
gfeaMap2 = []
for i in range(256):
    gfeaMap2.extend(gconv2_transpose[i][0])
gf3 = sess.run(g3, feed_dict={inputs_real: input_image,inputs_noise: batch_noise})
gconv3_transpose = sess.run(tf.transpose(gf3, [3, 0, 1, 2]))
gfeaMap3 = []
for i in range(128):
    gfeaMap3.extend(gconv2_transpose[i][0])
df2 = sess.run(d2, feed_dict={inputs_real: input_image,inputs_noise: batch_noise})
dconv2_transpose = sess.run(tf.transpose(df2, [3, 0, 1, 2]))
dfeaMap2 = []
for i in range(128):
    dfeaMap2.extend(dconv2_transpose[i][0])
df3 = sess.run(d3, feed_dict={inputs_real: input_image,inputs_noise: batch_noise})
dconv3_transpose = sess.run(tf.transpose(df3, [3, 0, 1, 2]))
dfeaMap3 = []
for i in range(256):
    dfeaMap3.extend(dconv3_transpose[i][0])
#-------------------------------------------------------prune----------------------
# gfeaMap2 gfeaMap3 dfeaMap2 dfeaMap3
Pruning_rate = 1/10
g2prune_list,g3prune_list,d2prune_list,d3prune_list =[]
for i in range(255):
    for j in range(255-i):
        vec1 = np.array(gfeaMap2[i])
        vec2 = np.array(gfeaMap2[j+1])
        distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
        g2prune_list.append(distance)
        vec3 = np.array(dfeaMap3[i])
        vec4 = np.array(dfeaMap3[j+1])
        distance2 = np.sqrt(np.sum(np.square(vec3 - vec4)))
        d3prune_list.append(distance2)
for i in range(127):
    for j in range(127-i):
        vec1 = np.array(gfeaMap3[i])
        vec2 = np.array(gfeaMap3[j+1])
        distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
        g3prune_list.append(distance)
        vec3 = np.array(dfeaMap2[i])
        vec4 = np.array(dfeaMap2[j+1])
        distance2 = np.sqrt(np.sum(np.square(vec3 - vec4)))
        d2prune_list.append(distance2)
gL2 = getListMaxNumIndex(g2prune_list, topk=len(g2prune_list)-len(g2prune_list)//(1/Pruning_rate))
gL3 = getListMaxNumIndex(g3prune_list, topk=len(g3prune_list)-len(g3prune_list)//(1/Pruning_rate))
dL2 = getListMaxNumIndex(d2prune_list, topk=len(d2prune_list)-len(d2prune_list)//(1/Pruning_rate))
dL3 = getListMaxNumIndex(d3prune_list, topk=len(d3prune_list)-len(d3prune_list)//(1/Pruning_rate))

gw2 = generator_variables_dict1['Gw2']
gb2 = generator_variables_dict1['Gb2']
gw3 = generator_variables_dict1['Gw3']
gb3 = generator_variables_dict1['Gb3']
gw4 = generator_variables_dict1['Gw4']
gb4 = generator_variables_dict1['Gb4']

dw1 = discriminator_variables_dict1['Dw1']
db1 = discriminator_variables_dict1['Db1']
dw2 = discriminator_variables_dict1['Dw2']
db2 = discriminator_variables_dict1['Db2']
dw3 = discriminator_variables_dict1['Dw3']
db3 = discriminator_variables_dict1['Db3']

gw2 = np.delete(gw2,gL2,axis=2)
gb2 = np.delete(gb2,gL2,axis=0)
gw3 = np.delete(gw3,gL3,axis=2)
gw3 = np.delete(gw3,gL3,axis=3)
gb3 = np.delete(gb3,gL3,axis=0)
gw4 = np.delete(gw4,gL3,axis=3)

dw1 = np.delete(dw1, dL2, axis=3)
db1 = np.delete(db1, dL2, axis=0)
dw2 = np.delete(dw2, dL3, axis=2)
dw2 = np.delete(dw2, dL3, axis=3)
db2 = np.delete(db2, dL3, axis=0)
dw3 = np.delete(dw3, dL3, axis=2)
generator_variables_dict2 = {
    "Gw2": tf.Variable(gw2, name='generator1/Gw2'),
    "Gb2": tf.Variable(gb2, name='generator1/Gb2'),
    "Gw3": tf.Variable(gw3, name='generator1/Gw3'),
    "Gb3": tf.Variable(gb3, name='generator1/Gb3'),
    "Gw4": tf.Variable(gw4, name='generator1/Gw4'),
    "Gb4": tf.Variable(gb4, name='generator1/Gb4')}
discriminator_variables_dict2 = {
    "Dw1": tf.Variable(dw1, name='discriminator1/Dw1'),
    "Db1": tf.Variable(db1, name='discriminator1/Db1'),
    "Dw2": tf.Variable(dw2, name='discriminator1/Dw2'),
    "Db2": tf.Variable(db2, name='discriminator1/Db2'),
    "Dw3": tf.Variable(dw3, name='discriminator1/Dw3'),
    "Db3": tf.Variable(db3, name='discriminator1/Db3')
}
def generator(noise_img, output_dim, is_train=True, alpha=0.01):
    with tf.variable_scope("generator1", reuse=(not is_train)):
        #64*100  to 64 4 x 4 x 512
        glayer1 = tf.layers.dense(noise_img, 4 * 4 * 512)
        glayer1 = tf.reshape(glayer1, [-1, 4, 4, 512])
        # batch normalization
        glayer1 = tf.layers.batch_normalization(glayer1, training=is_train)
        # Leaky ReLU
        glayer1 = tf.maximum(alpha * glayer1, glayer1)
        # dropout
        glayer1 = tf.nn.dropout(glayer1, keep_prob=0.8)

        # 4 x 4 x 512 to 8 x 8 x 256
        glayer2 = tf.nn.conv2d_transpose(glayer1,generator_variables_dict2["Gw2"],
                                        output_shape=tf.stack([64, 8, 8, 256]), strides=[1, 2, 2, 1],padding='SAME')
        glayer2 = tf.nn.bias_add(glayer2, generator_variables_dict2["Gb2"])
        glayer2 = tf.layers.batch_normalization(glayer2, training=is_train)
        glayer2 = tf.maximum(alpha * glayer2, glayer2)
        glayer2 = tf.nn.dropout(glayer2, keep_prob=0.8)

        # 8 x 8 256 to 16 x 16 x 128
        glayer3 = tf.nn.conv2d_transpose(glayer2, generator_variables_dict2["Gw3"],
                                        output_shape=tf.stack([64, 16, 16, 128]), strides=[1, 2, 2, 1], padding='SAME')
        glayer3 = tf.nn.bias_add(glayer3, generator_variables_dict2["Gb3"])
        # layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        glayer3 = tf.layers.batch_normalization(glayer3, training=is_train)
        glayer3 = tf.maximum(alpha * glayer3, glayer3)
        glayer3 = tf.nn.dropout(glayer3, keep_prob=0.8)

        # 16 x 16 x 128 to 32 x 32 x 3
        glayer4 = tf.nn.conv2d_transpose(glayer3, generator_variables_dict2["Gw4"],
                                        output_shape=tf.stack([64, 32, 32, output_dim]), strides=[1, 2, 2, 1], padding='SAME')
        glayer4 = tf.nn.bias_add(glayer4, generator_variables_dict2["Gb4"])
        # logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='same')
        outputs = tf.tanh(glayer4)
        return outputs
def discriminator(inputs_img, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator1", reuse=reuse):
        # 32 x 32 x 3 to 16 x 16 x 128
        dlayer1 = tf.nn.conv2d(inputs_img,discriminator_variables_dict2["Dw1"], strides=[1, 2, 2, 1], padding='SAME')
        dlayer1 = tf.nn.bias_add(dlayer1, discriminator_variables_dict2["Db1"])
        # layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides=2, padding='same')
        dlayer1 = tf.maximum(alpha * dlayer1, dlayer1)
        dlayer1 = tf.nn.dropout(dlayer1, keep_prob=0.8)

        # 16 x 16 x 128 to 8 x 8 x 256
        dlayer2 = tf.nn.conv2d(dlayer1, discriminator_variables_dict2["Dw2"], strides=[1, 2, 2, 1], padding='SAME')
        dlayer2 = tf.nn.bias_add(dlayer2, discriminator_variables_dict2["Db2"])
        # layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        dlayer2 = tf.layers.batch_normalization(dlayer2, training=True)
        dlayer2 = tf.maximum(alpha * dlayer2, dlayer2)
        dlayer2 = tf.nn.dropout(dlayer2, keep_prob=0.8)

        # 8 x 8 x 256 to 4 x 4 x 512
        dlayer3 = tf.nn.conv2d(dlayer2, discriminator_variables_dict2["Dw3"], strides=[1, 2, 2, 1], padding='SAME')
        dlayer3 = tf.nn.bias_add(dlayer3, discriminator_variables_dict2["Db3"])
        # layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        dlayer3 = tf.layers.batch_normalization(dlayer3, training=True)
        dlayer3 = tf.maximum(alpha * dlayer3, dlayer3)
        dlayer3 = tf.nn.dropout(dlayer3, keep_prob=0.8)

        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(dlayer3, (-1, 4 * 4 * 512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs
batch_size = 64
noise_size = 100
epochs = 50
n_samples = 64
learning_rate = 0.001
beta1 = 0.4
data_shape = [64,32,32,3]
losses = []
steps = 0

inputs_real = tf.placeholder(tf.float32, [None, data_shape[1], data_shape[2], data_shape[3]], name='inputs_real')
inputs_noise = tf.placeholder(tf.float32, [None, noise_size], name='inputs_noise')
# g_loss, d_loss = get_loss(inputs_real, inputs_noise, data_shape[-1])

g_outputs = get_generator(inputs_noise, data_shape[-1], is_train=True)
d_logits_real, d_outputs_real = get_discriminator(inputs_real)
d_logits_fake, d_outputs_fake = get_discriminator(g_outputs, reuse=True)
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_outputs_fake) * (1 - 0.1)))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_outputs_real) * (
                                                                                 1 - 0.1)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_outputs_fake)))
d_loss = tf.add(d_loss_real, d_loss_fake)
# g_train_opt, d_train_opt = get_optimizer(g_loss, d_loss, beta1, learning_rate)

train_vars = tf.trainable_variables()

g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# Optimizer
# with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(10):
        for batch_i in range(images.shape[0] // batch_size - 1):
            steps += 1
            batch_images = images[batch_i * batch_size: (batch_i + 1) * batch_size]

            # scale to -1, 1
            batch_images = batch_images * 2 - 1

            # noise
            batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))

            # run optimizer
            _ = sess.run(g_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})
            _ = sess.run(d_train_opt, feed_dict={inputs_real: batch_images,
                                                     inputs_noise: batch_noise})

            if steps % 10 == 0:
                train_loss_d = d_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                train_loss_g = g_loss.eval({inputs_real: batch_images,
                                                inputs_noise: batch_noise})
                losses.append((train_loss_d, train_loss_g))
                # 显示图片
                samples = show_generator_output(sess, n_samples, inputs_noise, data_shape[-1])
                plot_images(samples,steps)
                print("Epoch {}/{}".format(e + 1, 10),
                          "Discriminator Loss: {:.4f}".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))