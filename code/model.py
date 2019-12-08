import tensorflow as tf

def classifier_and_discriminator(input_var, name='C_D', reuse=None, NUM_CLASSES=1):
    with tf.variable_scope(name,reuse=reuse):
        leakyrelu_alpha = 0.2
        num_blocks = 5
        filters = 64
        kernel_size = 4
        strides = 2
        # Five intermediate blocks : conv + layer norm + instance norm + leaky relu
        for i in range(num_blocks):

            conv = tf.layers.conv2d(inputs = input_var,
                                    filters = filters,
                                    kernel_size = kernel_size,
                                    padding = 'same',
                                    #kernel_initializer= tf.contrib.layers.xavier_initializer(),
                                    strides = strides)
            layer_norm = tf.contrib.layers.layer_norm(conv)
            leaky_relu_out = tf.nn.leaky_relu(layer_norm, alpha = leakyrelu_alpha)

            input_var = leaky_relu_out
            filters += filters

        ### CLASSIFIER PART
        # Output block : fc(1024) + LN + leaky relu + fc(1)
        flatten_c = tf.contrib.layers.flatten(input_var)
        fc_c = tf.contrib.layers.fully_connected(flatten_c, num_outputs = 1024, activation_fn=None)
        #dropout_c = tf.nn.dropout(fc_c, 0.2)
        #layer_norm_c = tf.contrib.layers.layer_norm(dropout_c)
        #leaky_relu_out_c = tf.nn.leaky_relu(layer_norm_c, alpha = leakyrelu_alpha)
        leaky_relu_out_c = tf.nn.leaky_relu(fc_c, alpha = leakyrelu_alpha)
        # Classifier output
        flatten_out_c = tf.contrib.layers.flatten(leaky_relu_out_c)
        out_classifier = tf.contrib.layers.fully_connected(flatten_out_c,num_outputs=NUM_CLASSES, activation_fn=None)
        out_classifier = tf.nn.sigmoid(out_classifier)

        ### DISCRIMINATOR PART
        # Output block : fc(1024) + LN + leaky relu + fc(NUM_CLASS)
        flatten_d = tf.contrib.layers.flatten(input_var)
        fc_d = tf.contrib.layers.fully_connected(flatten_d, num_outputs = 1024, activation_fn=None)
        dropout_d = tf.nn.dropout(fc_d, 0.2)
        layer_norm_d = tf.contrib.layers.layer_norm(dropout_d)
        leaky_relu_out_d = tf.nn.leaky_relu(layer_norm_d, alpha = leakyrelu_alpha)
        # Classifier output
        flatten_out_d = tf.contrib.layers.flatten(leaky_relu_out_d)
        out_discriminator = tf.contrib.layers.fully_connected(flatten_out_d,num_outputs=1, activation_fn=None)

        return out_discriminator, out_classifier


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.002, name="conv2d", padding = 'SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer= tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.002))

        return tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())


def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.002, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer= tf.contrib.layers.xavier_initializer(),)

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.002))

        return tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

def encoder(inputs, name = 'G_encoder', reuse=tf.compat.v1.AUTO_REUSE, is_training = True):
    """
    encoder function
    :param: inputs
    :param: name
    :return list of layers:
    """

    with tf.variable_scope(name, reuse=reuse):

        leakyrelu_alpha = 0.2
        num_blocks = 5
        filters = 64
        kernel_size = 4
        strides = 2

        layers = []
        layers.append(inputs)

        for i in range(num_blocks):
            conv = conv2d(inputs, filters, kernel_size, kernel_size, strides, strides, name = str(i+1))
            batch_norm = tf.contrib.layers.layer_norm(conv)
            leaky_relu = tf.nn.leaky_relu(batch_norm, alpha = leakyrelu_alpha)

            inputs = leaky_relu
            filters += filters
            layers.append(inputs)

        return layers


def decoder(inputs, label, name = 'G_decoder', reuse=None, is_training = True):
    """
    decoder function
    :param: inputs (list of layers from encoder)
    :param: name
    :return tanh(conv5):
    """
    leakyrelu_alpha = 0.2
    filters = 1024
    kernel_size = 4
    strides = 2

    input_ = inputs[-1]

    def _attribute_concat(label, z):
        label = tf.expand_dims(label, 1)
        label = tf.expand_dims(label, 1)
        #label = label[:,tf.newaxis, tf.newaxis,:] #or use expand_dims twice
        label = tf.tile(label, [1, *z.get_shape().as_list()[1:3], 1])
        label = tf.cast(label, dtype=tf.float32)
        label = tf.concat([z, label], axis=3)
        return label

    input_ = _attribute_concat(label, input_)

    with tf.variable_scope(name, reuse=reuse):

        for ind in list(reversed(range(len(inputs)))):
            outout_shape = inputs[ind-1].get_shape().as_list()

            if ind==1:
                deconv = deconv2d(input_, outout_shape, kernel_size, kernel_size, strides, strides, name = "deconv_{}".format(ind))
                #deconv = tf.nn.conv2d_transpose(input_,  output_shape=outout_shape, strides=[1, 2, 2, 1])
                return tf.nn.tanh(deconv)

            deconv = deconv2d(input_, outout_shape, kernel_size, kernel_size, strides, strides, name = str(ind-1))
            concatenated = tf.concat([deconv, inputs[ind-1]], axis=3)

            batch_norm = tf.contrib.layers.layer_norm(concatenated)

            input_ = leaky_relu = tf.nn.relu(batch_norm, name = "ReLU_{}".format(ind))


def gradient_penalty(f, real, fake=None):
    def _interpolate(a, b=None):
        with tf.name_scope('interpolate'):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter
    with tf.name_scope('gradient_penalty'):
        x = _interpolate(real, fake)
        pred = f(x, reuse=tf.compat.v1.AUTO_REUSE)
        if isinstance(pred, tuple):
            pred = pred[0]
        grad = tf.gradients(pred, x)[0]
        norm = 1e-10 + tf.norm(tf.contrib.slim.flatten(grad), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
