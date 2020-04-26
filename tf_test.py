import tensorflow as tf

def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?

    #使用最近邻插值法对图片进行上采用
    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    return inputs

# https: // blog.csdn.net / Crystal_LYP / article / details / 103923101
# upsample1 = tf.image.resize_nearest_neighbor(encoded, (7,7))
# # Now 7x7x8
# conv4 = tf.layers.conv2d(upsample1, 8, (3,3), padding='same', activation=tf.nn.relu)
# # Now 7x7x8
# upsample2 = tf.image.resize_nearest_neighbor(conv4, (14,14))
# # Now 14x14x8
# conv5 = tf.layers.conv2d(upsample2, 8, (3,3), padding='same', activation=tf.nn.relu)
# # Now 14x14x8
# upsample3 = tf.image.resize_nearest_neighbor(conv5, (28,28))
# # Now 28x28x8
# conv6 = tf.layers.conv2d(upsample3, 16, (3,3), padding='same', activation=tf.nn.relu)
# # Now 28x28x16

output_channels = 10
input = tf.Variable(tf.random_normal([1, 28, 28, 3]))
filter = tf.Variable(tf.random_normal([3, 3, 3, output_channels]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    res = (sess.run(op))
    print (res.shape)
    # (1, 28, 28, output_channels) when padding='SAME'
    # (1, 26, 26, output_channels) when padding='VALID'
