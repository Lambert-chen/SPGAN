"""
@Time:2019.5.6
@Author: ChenYao
--------------------
:get_generator: generative model
:get_discriminator: discriminator model
"""
import tensorflow as tf
generator_variables_dict = {
    "Gw2": tf.Variable(tf.truncated_normal([4, 4, 32, 512], stddev=0.02), name='generator/Gw2'),
    "Gb2": tf.Variable(tf.constant(0.0, shape=[32]), name='generator/Gb2'),
    "Gw3": tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.02), name='generator/Gw3'),
    "Gb3": tf.Variable(tf.constant(0.0, shape=[16]), name='generator/Gb3'),
    "Gw4": tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.02), name='generator/Gw4'),
    "Gb4": tf.Variable(tf.constant(0.0, shape=[3]), name='generator/Gb4')
}
def get_generator(noise_img, output_dim, is_train=True, alpha=0.01):
    with tf.variable_scope("generator", reuse=(not is_train)):
        #64*100  to 64 4 x 4 x 512
        layer1 = tf.layers.dense(noise_img, 4 * 4 * 512)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 512])
        # batch normalization
        layer1 = tf.layers.batch_normalization(layer1, training=is_train)
        # Leaky ReLU
        layer1 = tf.maximum(alpha * layer1, layer1)
        # dropout
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)
        # 4 x 4 x 512 to 8 x 8 x 256
        layer2 = tf.nn.conv2d_transpose(layer1,generator_variables_dict["Gw2"],
                                        output_shape=tf.stack([64, 8, 8, 32]), strides=[1, 2, 2, 1],padding='SAME')
        layer2 = tf.nn.bias_add(layer2, generator_variables_dict["Gb2"])
        layer2 = tf.layers.batch_normalization(layer2, training=is_train)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        # 8 x 8 256 to 16 x 16 x 128
        layer3 = tf.nn.conv2d_transpose(layer2, generator_variables_dict["Gw3"],
                                        output_shape=tf.stack([64, 16, 16, 16]), strides=[1, 2, 2, 1], padding='SAME')
        layer3 = tf.nn.bias_add(layer3, generator_variables_dict["Gb3"])
        # layer3 = tf.layers.conv2d_transpose(layer2, 128, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=is_train)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        # 16 x 16 x 128 to 32 x 32 x 3
        layer4 = tf.nn.conv2d_transpose(layer3, generator_variables_dict["Gw4"],
                                        output_shape=tf.stack([64, 32, 32, output_dim]), strides=[1, 2, 2, 1], padding='SAME')
        layer4 = tf.nn.bias_add(layer4, generator_variables_dict["Gb4"])
        # logits = tf.layers.conv2d_transpose(layer3, output_dim, 3, strides=2, padding='same')
        outputs = tf.tanh(layer4)
        return outputs
discriminator_variables_dict = {
    "Dw1": tf.Variable(tf.truncated_normal([3, 3, 3, 16], stddev=0.02), name='discriminator/Dw1'),
    "Db1": tf.Variable(tf.constant(0.0, shape=[16]), name='discriminator/Db1'),
    "Dw2": tf.Variable(tf.truncated_normal([3, 3, 16,32], stddev=0.02), name='discriminator/Dw2'),
    "Db2": tf.Variable(tf.constant(0.0, shape=[32]), name='discriminator/Db2'),
    "Dw3": tf.Variable(tf.truncated_normal([3, 3, 32, 512], stddev=0.02), name='discriminator/Dw3'),
    "Db3": tf.Variable(tf.constant(0.0, shape=[512]), name='discriminator/Db3')
}
def get_discriminator(inputs_img, reuse=False, alpha=0.01):

    with tf.variable_scope("discriminator", reuse=reuse):
        # 32 x 32 x 3 to 16 x 16 x 128
        layer1 = tf.nn.conv2d(inputs_img,discriminator_variables_dict["Dw1"], strides=[1, 2, 2, 1], padding='SAME')
        layer1 = tf.nn.bias_add(layer1, discriminator_variables_dict["Db1"])
        # layer1 = tf.layers.conv2d(inputs_img, 128, 3, strides=2, padding='same')
        layer1 = tf.maximum(alpha * layer1, layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob=0.8)

        # 16 x 16 x 128 to 8 x 8 x 256
        layer2 = tf.nn.conv2d(layer1, discriminator_variables_dict["Dw2"], strides=[1, 2, 2, 1], padding='SAME')
        layer2 = tf.nn.bias_add(layer2, discriminator_variables_dict["Db2"])
        # layer2 = tf.layers.conv2d(layer1, 256, 3, strides=2, padding='same')
        layer2 = tf.layers.batch_normalization(layer2, training=True)
        layer2 = tf.maximum(alpha * layer2, layer2)
        layer2 = tf.nn.dropout(layer2, keep_prob=0.8)

        # 8 x 8 x 256 to 4 x 4 x 512
        layer3 = tf.nn.conv2d(layer2, discriminator_variables_dict["Dw3"], strides=[1, 2, 2, 1], padding='SAME')
        layer3 = tf.nn.bias_add(layer3, discriminator_variables_dict["Db3"])
        # layer3 = tf.layers.conv2d(layer2, 512, 3, strides=2, padding='same')
        layer3 = tf.layers.batch_normalization(layer3, training=True)
        layer3 = tf.maximum(alpha * layer3, layer3)
        layer3 = tf.nn.dropout(layer3, keep_prob=0.8)

        # 4 x 4 x 512 to 4*4*512 x 1
        flatten = tf.reshape(layer3, (-1, 4 * 4 * 512))
        logits = tf.layers.dense(flatten, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs
