"""
@Author: ChenYao
--------------------
Calculate SSIM for two pictures

"""
import tensorflow as tf
import os
image1 = 'D:/CelebA/Img/11/'
image2 = 'D:/CelebA/Img/22/'
# aa = tf.constant(0,shape=[15,2700])
with tf.Session() as sess:
    for path1 in os.listdir(image1):
        for path2 in os.listdir(image2):
            print(path1)
            print(path2)
            read_file1 = tf.read_file(image1+path1)
            read_file2 = tf.read_file(image2+path2)
            im1 = tf.image.decode_png(read_file1)
            im2 = tf.image.decode_png(read_file2)
            ssim1 = tf.image.ssim(im1, im2, max_val=255)
            ssim2 = tf.image.ssim_multiscale(im1, im2, max_val=255)
            print(sess.run(ssim1))
            print(sess.run(ssim2))


        # with tf.Session() as sess:
        #     print(sess.run(ssim1))
        #     print(sess.run(ssim2))
# Compute SSIM over tf.float32 Tensors.
# im1 = tf.image.convert_image_dtype(im1, tf.float32)
# im2 = tf.image.convert_image_dtype(im2, tf.float32)
# ssim2 = tf.image.ssim(im1, im2, max_val=1.0)

