import tensorflow as tf

net_params = {}

def neural_net_image_input(image_shape, name='x'):
    n_input_1 = image_shape[0]
    n_input_2 = image_shape[1]
    n_input_3 = image_shape[2]
    return tf.placeholder(tf.float32,[None, n_input_1, n_input_2, n_input_3], name=name)


def neural_net_label_input(n_classes):
    return tf.placeholder(tf.float32, [None, n_classes], name='y')


def neural_net_keep_prob_input():
    return tf.placeholder(tf.float32, None, name='keep_prob')


def conv_net(x, keep_prob):

    #layer = conv2d_maxpool(x, 16, (4,4), (1,1), (2,2), (2,2))
    layer = create_convolution_layers(x)
    tf.nn.dropout(layer, keep_prob=keep_prob)

    #layer = flatten(layer)
    layer = tf.contrib.layers.flatten(layer)
    #layer = fully_conn(layer, 400)
    layer = tf.contrib.layers.fully_connected(layer, 2000)
    layer = tf.nn.dropout(layer, keep_prob)
    layer = tf.contrib.layers.fully_connected(layer, 1000)
    layer = tf.nn.dropout(layer, keep_prob)

    res = tf.contrib.layers.fully_connected(layer, 4, activation_fn=None)

    return res


def create_convolution_layers(X):
    #Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')

    #alex net
    # [227x227x3] INPUT
    # [55x55x96] CONV1: 96 11x11 filters at stride 4, pad 0
    # [27x27x96] MAX POOL1: 3x3 filters at stride 2
    # [27x27x96] NORM1: Normalization layer
    # [27x27x256] CONV2: 256 5x5 filters at stride 1, pad 2
    # [13x13x256] MAX POOL2: 3x3 filters at stride 2
    # [13x13x256] NORM2: Normalization layer
    # [13x13x384] CONV3: 384 3x3 filters at stride 1, pad 1
    # [13x13x384] CONV4: 384 3x3 filters at stride 1, pad 1
    # [13x13x256] CONV5: 256 3x3 filters at stride 1, pad 1
    # [6x6x256] MAX POOL3: 3x3 filters at stride 2
    # [4096] FC6: 4096 neurons
    # [4096] FC7: 4096 neurons
    # [1000] FC8: 1000 neurons (class scores)

    #252 x 189
    #nn = add_conv_relu_maxPool()
    nn = create_conv2d(X, 128, strides=[8,8], w_name='W1')
    nn = tf.nn.relu(nn, name='W1_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    nn = create_conv2d(nn, 256, strides=[4,4], w_name='W2')
    nn = tf.nn.relu(nn, name='W2_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = create_conv2d(nn, 256, strides=[3,3], w_name='W3')
    nn = tf.nn.relu(nn, name='W3_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = create_conv2d(nn, 512, strides=[3,3], w_name='W4')
    nn = tf.nn.relu(nn, name='W4_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    nn = create_conv2d(nn, 512, strides=[3,3], w_name='W5')
    nn = tf.nn.relu(nn, name='W5_activated')
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    return nn

def add_conv_relu_maxPool(cnn, filters, strides, name):
    # nn = create_conv2d(nn, 512, strides=[3,3], w_name='W5')
    # nn = tf.nn.relu(nn)
    # nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    cnn = create_conv2d(cnn, filters, strides=strides, w_name=name)
    cnn = tf.nn.relu(cnn)
    return tf.nn.max_pool(cnn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def create_convolution_layers_ORIGINAL(X):
    #Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')

    Z1 = create_conv2d(X, 32, strides=[8,8], w_name='W1')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    Z2 = create_conv2d(P1, 64, strides=[4,4], w_name='W2')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    Z3 = create_conv2d(P2, 128, strides=[2,2], w_name='W3')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    return P3

def get_weights_shape(X, conv_num_outputs, strides):
    depth = X.get_shape().as_list()[-1]
    w_size = [strides[0], strides[1], depth, conv_num_outputs]
    c_strides = [1, strides[0], strides[1], 1]
    return w_size, c_strides

def create_conv2d(X, conv_num_outputs, strides, w_name):
    depth = X.get_shape().as_list()[-1]
    w_size = [strides[0], strides[1], depth, conv_num_outputs]
    c_strides = [1, strides[0], strides[1], 1]
    W = tf.get_variable(w_name, w_size, initializer=tf.contrib.layers.xavier_initializer(seed=0))
    Z = tf.nn.conv2d(X, W, strides=c_strides, padding='SAME', name=w_name+'_conv2d')
    return Z

def create_nn_conv2d(X, W, strides, w_name=''):
    c_strides = [1, strides[0], strides[1], 1]
    return tf.nn.conv2d(X, W, strides=c_strides, padding='SAME')


def create_weight_variables(shape, seed, name, use_gpu=False):
    if len(shape) == 4:
        in_out = shape[0] * shape[1] * shape[2] + shape[3]
    else:
        in_out = shape[0] + shape[1]

    import math
    stddev = math.sqrt(3.0 / in_out) # XAVIER INITIALIZER (GAUSSIAN)

    initializer = tf.truncated_normal(shape, stddev=stddev, seed=seed)

    if use_gpu:
        with tf.device("/gpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    else:
        with tf.device("/cpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)

def create_bias_variables(shape, name, use_gpu=False):
    initializer = tf.constant(0.1, shape=shape)

    if use_gpu:
        with tf.device("/gpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)
    else:
        with tf.device("/cpu"):
            return tf.get_variable(name, initializer=initializer, dtype=tf.float32)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

    depth = x_tensor.get_shape().as_list()[-1]
    bias = tf.Variable(tf.zeros(conv_num_outputs))

    c_strides = [1, conv_strides[0], conv_strides[1], 1]
    p_ksize = [1, pool_ksize[0], pool_ksize[1], 1]
    p_strides = [1, pool_strides[0], pool_strides[1], 1]

    # 2x2x5x10
    weight= tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], depth, conv_num_outputs]))


    conv = tf.nn.conv2d(x_tensor, weight, c_strides, 'SAME') + bias
    conv = tf.nn.relu(conv)

    pool = tf.nn.max_pool(conv, p_ksize, p_strides, 'SAME')

    return pool

def flatten(x_tensor):
    b, w, h, d = x_tensor.get_shape().as_list()
    img_size = w * h * d
    return tf.reshape(x_tensor, [-1, img_size])

def fully_conn(x_tensor, num_outputs):
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs]))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.nn.relu(tf.add(tf.matmul(x_tensor, weight), bias))

def output(x_tensor, num_outputs):
    shape = x_tensor.get_shape().as_list()
    weight = tf.Variable(tf.truncated_normal([shape[-1], num_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros(num_outputs))
    return tf.add(tf.matmul(x_tensor, weight), bias)

def get_bias(n):
    return tf.Variable(tf.zeros(n))

W1 = None
W2 = None
W3 = None
W4 = None
b1 = None
b2 = None
b3 = None
b4 = None

def init(use_gpu=0):
    global W1, W2, W3, W4, b1, b2, b3, b4

    with tf.variable_scope('weights') as scope:
        shape = get_weight_shape(w=16, h=16, depth=3, out=128)
        W1 = create_weight_variables(shape, seed=0, name="W_conv1", use_gpu=use_gpu)
        b1 = create_bias_variables([128], name="b_conv1", use_gpu=use_gpu)

        shape = get_weight_shape(w=8, h=8, depth=128, out=256)
        W2 = create_weight_variables(shape, seed=0, name="W_conv2", use_gpu=use_gpu)
        b2 = create_bias_variables([256], name="b_conv2", use_gpu=use_gpu)

        shape = get_weight_shape(w=4, h=4, depth=256, out=256)
        W3 = create_weight_variables(shape, seed=0, name="W_conv3", use_gpu=use_gpu)
        b3 = create_bias_variables([256], name="b_conv3", use_gpu=use_gpu)

        shape = get_weight_shape(w=4, h=4, depth=256, out=512)
        W4 = create_weight_variables(shape, seed=0, name="W_conv4", use_gpu=use_gpu)
        b4 = create_bias_variables([512], name="b_conv4", use_gpu=use_gpu)

def get_weight_shape(w, h, depth, out):
    return [w, h, depth, out]

def make_logits(tensor, keep_prob, reuse=True):

    with tf.variable_scope("weights", reuse=True): W1 = tf.get_variable("W_conv1")
    with tf.variable_scope("weights", reuse=True): W2 = tf.get_variable("W_conv2")
    with tf.variable_scope("weights", reuse=True): W3 = tf.get_variable("W_conv3")
    with tf.variable_scope("weights", reuse=True): W4 = tf.get_variable("W_conv4")

    with tf.variable_scope("weights", reuse=True): b1 = tf.get_variable("b_conv1")
    with tf.variable_scope("weights", reuse=True): b2 = tf.get_variable("b_conv2")
    with tf.variable_scope("weights", reuse=True): b3 = tf.get_variable("b_conv3")
    with tf.variable_scope("weights", reuse=True): b4 = tf.get_variable("b_conv4")

    with tf.name_scope('conv2d_1') as scope:
        nn = create_nn_conv2d(tensor, W1, [16, 16])
    nn = tf.nn.tanh(tf.nn.bias_add(nn, b1))
    nn = tf.layers.batch_normalization(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    with tf.name_scope('conv2d_2') as scope:
        nn = create_nn_conv2d(nn, W2, [8, 8])
    nn = tf.nn.tanh(tf.nn.bias_add(nn, b2))
    nn = tf.layers.batch_normalization(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    with tf.name_scope('conv2d_3') as scope:
        nn = create_nn_conv2d(nn, W3, [4, 4])
    nn = tf.nn.tanh(tf.nn.bias_add(nn, b3))
    nn = tf.layers.batch_normalization(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    with tf.name_scope('conv2d_4') as scope:
        nn = create_nn_conv2d(nn, W4, [4, 4])
    nn = tf.nn.tanh(tf.nn.bias_add(nn, b4))
    nn = tf.layers.batch_normalization(nn)
    nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

    tf.nn.dropout(nn, keep_prob=keep_prob)

    layer = tf.contrib.layers.fully_connected(nn, 1024, activation_fn=None, scope='Bottleneck', reuse=reuse)
    #layer = tf.contrib.nn.fully_connected(nn, 1024, activation_fn=tf.nn.sigmoid, scope = 'fc1')
    #tf.nn.dropout(layer, keep_prob=keep_prob)
    #layer = tf.contrib.layers.fully_connected(layer, 1024, activation_fn=tf.nn.tanh)
    #tf.nn.dropout(layer, keep_prob=keep_prob)
    #layer = tf.contrib.layers.fully_connected(layer, 512, activation_fn=None)
    return layer

def make_logits2(tensor, keep_prob, reuse=True):

    # Model
    with tf.variable_scope('weights', reuse=reuse) as scope:
        nn = create_conv2d(tensor, 128, strides=[32, 32], w_name='W1')
        nn = tf.nn.relu(nn)
        nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        nn = create_conv2d(nn, 256, strides=[16, 16], w_name='W2')
        nn = tf.nn.relu(nn)
        nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

        nn = create_conv2d(nn, 256, strides=[8, 8], w_name='W3')
        nn = tf.nn.relu(nn)
        nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

        nn = create_conv2d(nn, 512, strides=[8, 8], w_name='W4')
        nn = tf.nn.relu(nn)
        nn = tf.nn.max_pool(nn, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

        tf.nn.dropout(nn, keep_prob=keep_prob)
        layer = tf.contrib.layers.flatten(nn)
        layer = tf.contrib.layers.fully_connected(layer, 1024, activation_fn=None)

    return layer

class ConvolutionalBatchNormalizer(object):
    """Helper class that groups the normalization logic and variables.

    Use:
        ewma = tf.train.ExponentialMovingAverage(decay=0.99)
        bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)
        update_assignments = bn.get_assigner()
        x = bn.normalize(y, train=training?)
        (the output x will be batch-normalized).
    """

    def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                                trainable=False)
        self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                    trainable=False)
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
        self.ewma_trainer = ewma_trainer
        self.epsilon = epsilon
        self.scale_after_norm = scale_after_norm

    def get_assigner(self):
        """Returns an EWMA apply op that must be invoked after optimization."""
        return self.ewma_trainer.apply([self.mean, self.variance])

    def normalize(self, x, train=True):
        """Returns a batch-normalized version of x."""
        if train:
            mean, variance = tf.nn.moments(x, [0, 1, 2])
            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_norm_with_global_normalization(
                    x, mean, variance, self.beta, self.gamma,
                    self.epsilon, self.scale_after_norm)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, local_beta, local_gamma,
                self.epsilon, self.scale_after_norm)