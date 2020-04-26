import tensorflow as tf
"""
Reference website:
https://blog.csdn.net/litt1e/article/details/88907542
Yolo_v3使用了darknet-53的前面的52层（没有全连接层），yolo_v3这个网络是一个全卷积网络，大量使用残差的跳层连接，并且为了降低池化带来的梯度负面效果。
作者直接摒弃了POOLing，用步长为2的卷积conv的stride来实现降采样。

"""

def conv2d(inputs, filters, kernel_size, strides=1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size)
    inputs = tf.contrib.slim.conv2d(inputs,
                                    filters,
                                    kernel_size,
                                    stride=strides,
                                    padding=('SAME' if strides == 1 else 'VALID')
                                    )
    return inputs


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    # inputs 4个维度, 只在第2，3维度扩充
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
    return padded_inputs


"""
ResNet 思想
Origin:
z(l + 1) = W(l + 1) * a(l) + b(l + 1)
a(l + 1) = g[z(l + 1)] 

ResNet:
a(l + 2) = g[z(l + 2) + a(l)] 
"""
def res_unit(inputs, filters):

    short_cut = inputs

    net = conv2d(inputs, filters * 1, 1)

    net = conv2d(net, filters * 2, 3)

    net = net + short_cut

    return net


def darknet53(inputs):
    # first two conv2d layers

    net = conv2d(inputs, 32, 3, strides=1) # 1, map_size=(w,h) (1, 416, 416, 3)

    net = conv2d(net, 64, 3, strides=2)  # 2, map size=(w/2,h/2) (1, 208, 208, 64)

    # res_block * 1
    net = res_unit(net, 32)  # 3,4 (1, 208, 208, 64)

    net = conv2d(net, 128, 3, strides=2)  # 5, map size=(w/4,h/4) (1, 104, 104, 128)

    # res_block * 2
    for i in range(2):
        net = res_unit(net, 64)  # 6,7,8,9

    net = conv2d(net, 256, 3, strides=2)  # 10, map size=(w/8,h/8) (1, 52, 52, 256)

    # res_block * 8
    for i in range(8):
        net = res_unit(net, 128)  # 11-26

    route_1 = net  # 26
    net = conv2d(net, 512, 3, strides=2)  # 27, map size=(w/16,h/16) (1, 26, 26, 512)

    # res_block * 8
    for i in range(8):
        net = res_unit(net, 256)  # 28-43

    route_2 = net  # 43
    net = conv2d(net, 1024, 3, strides=2)  # 44, map size=(w/32,h/32) (1, 13, 13, 512)

    # res_block * 4
    for i in range(4):
        net = res_unit(net, 512)  # 45-52
    route_3 = net  # 52

    return route_1, route_2, route_3 # 8倍, 16倍, 32倍 downsample


def test_darknet53_shape():
    inputs = tf.ones(shape=[1, 416, 416, 3])
    # # r1 = darknet53_body(input)
    # # r1 = conv2d(inputs, 32, 3, strides=1)
    # outputs = tf.contrib.slim.conv2d(inputs, 32, [3, 3], stride=1, weights_initializer=tf.ones_initializer, padding='SAME')
    # y2 = tf.contrib.slim.conv2d(inputs, 64, [5, 5], weights_initializer=tf.ones_initializer, padding='SAME')
    r1, r2 ,r3 = darknet53(inputs)
    with tf.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        input_value, route_1_value, route_2_value, route_3_value = sess.run([inputs, r1, r2 ,r3])
        print(route_1_value.shape) # (1, 52, 52, 256)
        print(route_2_value.shape) # (1, 26, 26, 512)
        print(route_3_value.shape) # (1, 13, 13, 1024)


test_darknet53_shape()


def test_tf_pad():
    t= [[2,3,4],[5,6,7]]
    t_pad = tf.pad(t,[[1,1],[2,2]], mode="CONSTANT")
    with tf.compat.v1.Session() as sess:
        print(sess.run(t_pad))
        # [[0 0 0 0 0 0 0]
        #  [0 0 2 3 4 0 0]
        #  [0 0 5 6 7 0 0]
        #  [0 0 0 0 0 0 0]]


def test_fixed_padding():
    for kernel_size in range(100):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        print(kernel_size, pad_beg, pad_end)

        # kernel_size = 2k,     pad_beg = k - 1, pad_end = k
        # kernel_size = 2k + 1, pad_beg = k    , pad_end = k


# def compare_tf_nn_conv2d_with_slim_conv2d():
# x1 = tf.ones(shape=[1, 416, 416, 3])
# y2 = darknet53_body(x1)
# weights_initializer用于指定权重的初始化程序
# weights_initializer = initializers.xavier_initializer(),

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     y2_value = sess.run(y2)
#     print(y2_value)




# test_tf_pad()
# test_fixed_padding()
# compare_tf_nn_conv2d_with_slim_conv2d()
