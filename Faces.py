import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import os

path = "./10kFaces/"

numFaces = 100

fileNames = []
images = []
arrays = []

faceFolder = os.listdir(path)

for i in range(0, numFaces):
    if (faceFolder[i][-4:] == ".jpg"):
        images.append(Image.open(""+path+faceFolder[i]))

for i in range(0, len(images)):
    images[i] = images[i].resize( (128, 128) )
    images[i] = images[i].convert("L")

for i in range(0, len(images)):
    images[i] = list(images[i].getdata())
    images[i] = np.array(images[i])
    images[i] = images[i].astype(np.float32)
    images[i] = images[i]/255.0
    images[i] = images[i].reshape((128, 128))

#The 'images' array now contains 128x128 numpy arrays of normalized greyscale values

def conv2d(x, W):
  return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')

def avg_pool_2x2(x):
  return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def discriminator(x_image, reuse=False):
  with tf.variable_scope('discriminator') as scope:
    if (reuse):
      tf.get_variable_scope().reuse_variables()
    #First Conv and Pool Layers
    W_conv1 = tf.get_variable("d_wconv1", [5,5,1,8], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv1 = tf.get_variable("d_bconv1", [8], initializer=tf.constant_initializer(0))
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = avg_pool_2x2(h_conv1)

    #Second Conv and Pool Layers
    W_conv2 = tf.get_variable("d_wconv2", [5,5,8,16], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_conv2 = tf.get_variable("d_bconv2", [16], initializer=tf.constant_initializer(0))
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = avg_pool_2x2(h_conv2)

    #First Fully-Connected Layer
    W_fc1 = tf.get_variable("d_wfc1", [7*7*16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_fc1 = tf.get_variable("d_bfc1", [32], initializer=tf.constant_initializer(0))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #Second Fully-Connected Layer
    W_fc2 = tf.get_variable("d_wfc2", [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_fc2 = tf.get_variable("d_bfc2", [1], initializer=tf.constant_initializer(0))

    #Final layer
    y_conv = (tf.matmul(h_fc1, W_fc2) + b_fc2)
  return y_conv

def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope('generator') as scope:
        if (reuse):
            tf.get_variable_scope().reuse_variables()
        #g_dim = 64 #Number of filters of first layer of generator 
        g_dim = 512
        c_dim = 1 #Color dimension of output (MNIST is grayscale, so c_dim = 1 for us)
        s = 128 #Output size of the image
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) #We want to slowly upscale the image, so these values will help
                                                                  #make that change gradual.

        h0 = tf.reshape(z, [batch_size, s16, s16, 256])
        h0 = tf.nn.relu(h0)
        #Dimensions of h0 = batch_size x 8 x 8 x 256

        #First DeConv Layer
        output1_shape = [batch_size, s8, s8, 64]
        W_conv1 = tf.get_variable('g_wconv1', [16, 16, output1_shape[-1], int(h0.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv1
        H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
        H_conv1 = tf.nn.relu(H_conv1)
        #Dimensions of H_conv1 = batch_size x 16 x 16 x 64

        #Second DeConv Layer
        output2_shape = [batch_size, s4, s4 , 16]
        W_conv2 = tf.get_variable('g_wconv2', [16, 16, output2_shape[-1], int(H_conv1.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv2
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        H_conv2 = tf.nn.relu(H_conv2)
        #Dimensions of H_conv2 = batch_size x 32 x 32 x 16

        #Third DeConv Layer
        output3_shape = [batch_size, s2, s2, 4]
        W_conv3 = tf.get_variable('g_wconv3', [16, 16, output3_shape[-1], int(H_conv2.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, 
                                         strides=[1, 2, 2, 1], padding='SAME') + b_conv3
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)
        #Dimensions of H_conv3 = batch_size x 64 x 64 x 4

        #Fourth DeConv Layer
        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [16, 16, output4_shape[-1], int(H_conv3.get_shape()[-1])], 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, 
                                         strides=[1, 2, 2, 1], padding='VALID') + b_conv4
        H_conv4 = tf.nn.tanh(H_conv4)
        #Dimensions of H_conv4 = batch_size x 128 x 128 x 1

    return H_conv4

#magic number: 16384
sess = tf.Session()
z_dimensions = 16384
z_test_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

sample_image = generator(z_test_placeholder, 1, z_dimensions)
test_z = np.random.normal(-1, 1, [1,z_dimensions])

sess.run(tf.global_variables_initializer())
temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))

