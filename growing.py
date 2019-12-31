"""
@Time:2019.5.6
@Author: ChenYao
--------------------
Network reconstruction after growth
"""
import tensorflow as tf
import numpy as np
from SPGAN.GandD import generator_variables_dict,discriminator_variables_dict
from SPGAN.load_data import load_cfar10_batch
cifar10_path = 'cifar-10-python\cifar-10-batches-py'
x_train, y_train = load_cfar10_batch(cifar10_path, 1)
for i in range(2, 6):
    features, labels = load_cfar10_batch(cifar10_path, i)
    x_train, y_train = np.concatenate([x_train, features]), np.concatenate([y_train, labels])

gw2 = generator_variables_dict['Gw2']
gb2 = generator_variables_dict['Gb2']
gw3 = generator_variables_dict['Gw3']
gb3 = generator_variables_dict['Gb3']
gw4 = generator_variables_dict['Gw4']
gb4 = generator_variables_dict['Gb4']

dw1 = discriminator_variables_dict['Dw1']
db1 = discriminator_variables_dict['Db1']
dw2 = discriminator_variables_dict['Dw2']
db2 = discriminator_variables_dict['Db2']
dw3 = discriminator_variables_dict['Dw3']
db3 = discriminator_variables_dict['Db3']
#growing:16----->128   32---->256
for i in range(3):
    gw2 = tf.concat([gw2,gw2],2)
    gb2 = tf.concat([gb2,gb2],0)
    gw3 = tf.concat([gw3,gw3],2)
    gw3 = tf.concat([gw3,gw3],3)
    gb3 = tf.concat([gb3,gb3],0)
    gw4 = tf.concat([gw4, gw4], 3)

    dw1 = tf.concat([dw1,dw1], 3)
    db1 = tf.concat([db1,db1],0)
    dw2 = tf.concat([dw2,dw2], 2)
    dw2 = tf.concat([dw2,dw2], 3)
    db2 = tf.concat([db2,db2],0)
    dw3 = tf.concat([dw3,dw3], 2)
generator_variables_dict1 = {
    "Gw2": tf.Variable(gw2, name='generator1/Gw2'),
    "Gb2": tf.Variable(gb2, name='generator1/Gb2'),
    "Gw3": tf.Variable(gw3, name='generator1/Gw3'),
    "Gb3": tf.Variable(gb3, name='generator1/Gb3'),
    "Gw4": tf.Variable(gw4, name='generator1/Gw4'),
    "Gb4": tf.Variable(gb4, name='generator1/Gb4')
}
g2 = []
g3 = []
d2 = []
d3 = []

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
        glayer2 = tf.nn.conv2d_transpose(glayer1,generator_variables_dict1["Gw2"],
                                        output_shape=tf.stack([64, 8, 8, 256]), strides=[1, 2, 2, 1],padding='SAME')
        glayer2 = tf.nn.bias_add(glayer2, generator_variables_dict1["Gb2"])
        glayer2 = tf.layers.batch_normalization(glayer2, training=is_train)
        glayer2 = tf.maximum(alpha * glayer2, glayer2)
        glayer2 = tf.nn.dropout(glayer2, keep_prob=0.8)
        g2 = glayer2

        # 8 x 8 256 to 16 x 16 x 128
        glayer3 = tf.nn.conv2d_transpose(glayer2, generator_variables_dict1["Gw3"],
                                        output_shape=tf.stack([64, 16, 16, 128]), strides=[1, 2, 2, 1], padding='SAME')
        glayer3 = tf.nn.bias_add(glayer3, generator_variables_dict1["Gb3"])
        # layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        glayer3 = tf.layers.batch_normalization(glayer3, training=is_train)
        glayer3 = tf.maximum(alpha * glayer3, glayer3)
        glayer3 = tf.nn.dropout(glayer3, keep_prob=0.8)
        g3 = glayer3

        # 16 x 16 x 128 to 32 x 32 x 3
        glayer4 = tf.nn.conv2d_transpose(glayer3, generator_variables_dict1["Gw4"],
                                        output_shape=tf.stack([64, 32, 32, output_dim]), strides=[1, 2, 2, 1], padding='SAME')
        glayer4 = tf.nn.bias_add(glayer4, generator_variables_dict1["Gb4"])
        # logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='same')
        outputs = tf.tanh(glayer4)
        return outputs
discriminator_variables_dict1 = {
    "Dw1": tf.Variable(dw1, name='discriminator1/Dw1'),
    "Db1": tf.Variable(db1, name='discriminator1/Db1'),
    "Dw2": tf.Variable(dw2, name='discriminator1/Dw2'),
    "Db2": tf.Variable(db2, name='discriminator1/Db2'),
    "Dw3": tf.Variable(dw3, name='discriminator1/Dw3'),
    "Db3": tf.Variable(db3, name='discriminator1/Db3')
}
def discriminator(inputs_img, reuse=False, alpha=0.01):
    with tf.variable_scope("discriminator1", reuse=reuse):
        # 32 x 32 x 3 to 16 x 16 x 128
        dlayer1 = tf.nn.conv2d(inputs_img,discriminator_variables_dict1["Dw1"], strides=[1, 2, 2, 1], padding='SAME')
        dlayer1 = tf.nn.bias_add(dlayer1, discriminator_variables_dict1["Db1"])
        # layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides=2, padding='same')
        dlayer1 = tf.maximum(alpha * dlayer1, dlayer1)
        dlayer1 = tf.nn.dropout(dlayer1, keep_prob=0.8)

        # 16 x 16 x 128 to 8 x 8 x 256
        dlayer2 = tf.nn.conv2d(dlayer1, discriminator_variables_dict1["Dw2"], strides=[1, 2, 2, 1], padding='SAME')
        dlayer2 = tf.nn.bias_add(dlayer2, discriminator_variables_dict1["Db2"])
        # layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        dlayer2 = tf.layers.batch_normalization(dlayer2, training=True)
        dlayer2 = tf.maximum(alpha * dlayer2, dlayer2)
        dlayer2 = tf.nn.dropout(dlayer2, keep_prob=0.8)
        d3 = dlayer2

        # 8 x 8 x 256 to 4 x 4 x 512
        dlayer3 = tf.nn.conv2d(dlayer2, discriminator_variables_dict1["Dw3"], strides=[1, 2, 2, 1], padding='SAME')
        dlayer3 = tf.nn.bias_add(dlayer3, discriminator_variables_dict1["Db3"])
        # layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        dlayer3 = tf.layers.batch_normalization(dlayer3, training=True)
        dlayer3 = tf.maximum(alpha * dlayer3, dlayer3)
        dlayer3 = tf.nn.dropout(dlayer3, keep_prob=0.8)
        d3 = dlayer3

        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(dlayer3, (-1, 4 * 4 * 512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)
        return logits, outputs
def prune_feature_maps(input_image,batch_noise,inputs_real,inputs_noise):
    input_image = x_train[59:60]
    prune_list = input_image
    return prune_list
def feature_maps():
    return g2,g3 ,d2,d3