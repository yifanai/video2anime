from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


_EPSILON = 1e-5

class UGATIT(object):
    def generate(self, x, scope="generator_B", reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            channels = 64 
            x = conv(x, channels, scope="conv", kernel=7, pad=3)
            x = tf.contrib.layers.instance_norm(x, scope="ins_norm")
            x = tf.nn.relu(x)

            for i in range(2):
                x = conv(x, channels*2, scope="conv_" + str(i), stride=2)
                x = tf.contrib.layers.instance_norm(x, scope="ins_norm_" + str(i))
                x = tf.nn.relu(x)
                channels *= 2

            for i in range(4):
                x = resblock(x, channels, scope='resblock_' + str(i))

            cam_x = tf.keras.layers.GlobalAveragePooling2D()(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, scope='CAM_logit')
            x_gap = x * cam_x_weight

            cam_x = tf.keras.layers.GlobalMaxPool2D()(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, reuse=True, scope='CAM_logit')
            x_gmp = x * cam_x_weight

            x = tf.concat([x_gap, x_gmp], axis=-1)
            x = conv(x, channels, kernel=1, pad=0, pad_mode="CONSTANT", scope="conv_1x1")
            x = tf.nn.relu(x)

            gamma, beta = self.MLP(x)

            for i in range(4):
                x = adaptive_ins_layer_resblock(x, channels, gamma, beta, scope='adaptive_resblock' + str(i))

            for i in range(2) :
                x = tf.keras.layers.UpSampling2D()(x) 
                x = conv(x, channels // 2, scope='up_conv_' + str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm_' + str(i))
                x = tf.nn.relu(x)
                channels //= 2
            x = conv(x, channels=3, scope='G_logit', kernel=7, pad=3)
            x = tf.nn.tanh(x)
            return x

    def MLP(self, x, scope='MLP'):
        with tf.variable_scope(scope):
            units = 64 * 4
            for i in range(2) :
                x = fully_connected(x, units, scope='linear_' + str(i))
                x = tf.nn.relu(x)
            gamma = fully_connected(x, units, scope='gamma')
            gamma = tf.reshape(gamma, shape=[1, 1, 1, units])
            beta = fully_connected(x, units, scope='beta')
            beta = tf.reshape(beta, shape=[1, 1, 1, units])
            return gamma, beta


"""
Layers
"""
def conv(x, channels, scope="conv_0", kernel=3, stride=1, pad=1, pad_mode="REFLECT"):
    with tf.variable_scope(scope):
        if pad > 0 :
            if (kernel - stride) % 2 == 0:
                top = bottom = left = right = pad
            else:
                top = left = pad
                bottom = right = kernel - stride - pad
            x = tf.pad(x, [[0, 0], [top, bottom], [left, right], [0, 0]], mode=pad_mode)
        return tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=kernel,
            strides=stride,
            name="conv2d"
        )(x)

def fully_connected(x, units, scope='linear'):
    with tf.variable_scope(scope):
        x = tf.keras.layers.Flatten()(x)
        return tf.keras.layers.Dense(units=units, name="dense")(x)

def fully_connected_with_w(x, scope='linear', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        x = tf.keras.layers.Flatten()(x)
        channels = x.shape[-1]
        w = tf.get_variable("kernel", [channels, 1])
        b = tf.get_variable("bias", [1])
        x = tf.nn.bias_add(tf.matmul(x, w), b)
        weights = tf.gather(tf.transpose(tf.nn.bias_add(w, b)), 0)
        return x, weights


"""
Blocks
"""
def resblock(x, channels, scope):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            y = conv(x, channels)
            y = tf.contrib.layers.instance_norm(y, scope="instance_norm")
            y = tf.nn.relu(y)
        with tf.variable_scope('res2'):
            y = conv(y, channels)
            y = tf.contrib.layers.instance_norm(y, scope="instance_norm")
        return x + y

def adaptive_ins_layer_resblock(x, channels, gamma, beta, scope='adaptive_resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            y = conv(x, channels)
            y = adaptive_instance_layer_norm(y, gamma, beta)
            y = tf.nn.relu(y)
        with tf.variable_scope('res2'):
            y = conv(y, channels)
            y = adaptive_instance_layer_norm(y, gamma, beta)
        return x + y 


"""
Normalization
"""
def layer_instance_norm(x, scope='layer_instance_norm'):
    with tf.variable_scope(scope):
        channels = x.shape[-1] 
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + _EPSILON))
        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + _EPSILON))
        rho = tf.get_variable("rho", [channels], constraint=lambda x: tf.clip_by_value(x, 0, 1))
        gamma = tf.get_variable("gamma", [channels])
        beta = tf.get_variable("beta", [channels])
        x_hat = rho * x_ins + (1 - rho) * x_ln
        x_hat = x_hat * gamma + beta
        return x_hat

def adaptive_instance_layer_norm(x, gamma, beta, scope='instance_layer_norm'):
    with tf.variable_scope(scope):
        channels = x.shape[-1]
        ins_mean, ins_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_ins = (x - ins_mean) / (tf.sqrt(ins_sigma + _EPSILON))
        ln_mean, ln_sigma = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
        x_ln = (x - ln_mean) / (tf.sqrt(ln_sigma + _EPSILON))
        rho = tf.get_variable("rho", [channels], constraint=lambda x: tf.clip_by_value(x, 0, 1))
        rho = tf.clip_by_value(rho - 0.1, 0, 1)
        x_hat = rho * x_ins + (1 - rho) * x_ln
        x_hat = x_hat * gamma + beta
        return x_hat
