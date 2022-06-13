import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from setup_cifar import CIFAR
import numpy as np
import time
import sys

part = 1

if len(sys.argv) > 1:
    part = int(sys.argv[1])


# Normal training
def train_normal(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                 act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    x = x0
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b
    layers.append(x)

    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Normal training with adam + standard parameterization
def train_normal_adam(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                      act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    x = x0
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T / np.sqrt(k * k * last_shape)
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T / np.sqrt(last_shape)
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') + b
    layers.append(x)

    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Flattens
def flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return [z for y in x for z in flatten(y)]
    else:
        return [x]


# Align-ada training
def train_alignada(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                   act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    x_lin = x0
    layers = [x]
    layers_lin = [x_lin]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            x_lin = tf.nn.conv2d(x_lin + x - tf.stop_gradient(x), W_lin, [1, s, s, 1], p) / \
                    np.sqrt(k * k * last_shape) + b_lin
        else:
            x_lin = tf.nn.conv2d(act(x_lin + x - tf.stop_gradient(x)), W_lin, [1, s, s, 1], p) / \
                    np.sqrt(k * k * last_shape) + b_lin
        if i == 0:
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        mean, var = tf.nn.moments(x_lin, [0])
        x_lin = tf.nn.batch_normalization(x_lin, mean, var, None, None, 0.000001)
        layers.append(x)
        layers_lin.append(x_lin)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
    x_lin = tf.nn.avg_pool(act(x_lin), [1, x_lin.shape[1], x_lin.shape[2], 1],
                           [1, x_lin.shape[1], x_lin.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    x_lin = tf.nn.conv2d(x_lin + x - tf.stop_gradient(x), W_lin, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b_lin
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b
    layers.append(x)

    logits = tf.layers.flatten(x)
    logits_lin = tf.layers.flatten(x_lin)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean(
        (logits + logits_lin - tf.stop_gradient(logits_lin) - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Align-ada training with adam + standard parameterization
def train_alignada_adam(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                        act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    x_lin = x0
    layers = [x]
    layers_lin = [x_lin]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T / np.sqrt(k * k * last_shape)
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            x_lin = tf.nn.conv2d(x_lin + x - tf.stop_gradient(x), W_lin, [1, s, s, 1], p) + b_lin
        else:
            x_lin = tf.nn.conv2d(act(x_lin + x - tf.stop_gradient(x)), W_lin, [1, s, s, 1], p) + b_lin
        if i == 0:
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) + b
        else:
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        mean, var = tf.nn.moments(x_lin, [0])
        x_lin = tf.nn.batch_normalization(x_lin, mean, var, None, None, 0.000001)
        layers.append(x)
        layers_lin.append(x_lin)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
    x_lin = tf.nn.avg_pool(act(x_lin), [1, x_lin.shape[1], x_lin.shape[2], 1],
                           [1, x_lin.shape[1], x_lin.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T / np.sqrt(last_shape)
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    x_lin = tf.nn.conv2d(x_lin + x - tf.stop_gradient(x), W_lin, [1, 1, 1, 1], 'VALID') + b_lin
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') + b
    layers.append(x)

    logits = tf.layers.flatten(x)
    logits_lin = tf.layers.flatten(x_lin)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean(
        (logits + logits_lin - tf.stop_gradient(logits_lin) - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Direct feedback alignment training
def train_dfa(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
              act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3
    cur_size = 32
    n_classes = 10

    x0 = inputs

    params = []
    x = x0
    out_grad = tf.zeros_like(labels)
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        if i == 0:
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        cur_size = cur_size // s
        B_val = np.random.normal(scale=1 / np.sqrt(cur_size * cur_size * l),
                                 size=(n_classes, cur_size * cur_size * l)).T
        B_val = B_val.reshape((cur_size, cur_size, l, n_classes))
        B = tf.Variable(tf.convert_to_tensor(B_val, dtype=tf.float32))
        proj = tf.layers.flatten(tf.nn.conv2d(act(x), B, [1, cur_size, cur_size, 1], 'VALID'))
        out_grad = out_grad + proj - tf.stop_gradient(proj)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b
    layers.append(x)
    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits + out_grad - labels) ** 2)
    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# DFA + adam, standard parameterization
def train_dfa_adam(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                   act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3
    cur_size = 32
    n_classes = 10

    x0 = inputs

    params = []
    x = x0
    out_grad = tf.zeros_like(labels)
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l)) / np.sqrt(k * k * last_shape)
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        if i == 0:
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) + b
        else:
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        cur_size = cur_size // s
        B_val = np.random.normal(scale=1 / np.sqrt(cur_size * cur_size * l),
                                 size=(n_classes, cur_size * cur_size * l)).T
        B_val = B_val.reshape((cur_size, cur_size, l, n_classes))
        B = tf.Variable(tf.convert_to_tensor(B_val, dtype=tf.float32))
        proj = tf.layers.flatten(tf.nn.conv2d(act(x), B, [1, cur_size, cur_size, 1], 'VALID'))
        out_grad = out_grad + proj - tf.stop_gradient(proj)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10)) / np.sqrt(last_shape)
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') + b
    layers.append(x)
    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits + out_grad - labels) ** 2)
    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Feedback alignment training
def train_fa(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
             act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            fa_grad = tf.nn.conv2d(x, W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape)
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) / np.sqrt(
                k * k * last_shape) + b + fa_grad - tf.stop_gradient(fa_grad)
        else:
            fa_grad = tf.nn.conv2d(act(x), W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape)
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) / np.sqrt(
                k * k * last_shape) + b + fa_grad - tf.stop_gradient(fa_grad)
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    fa_grad = tf.nn.conv2d(x, W_lin, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape)
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') / np.sqrt(
        last_shape) + b + fa_grad - tf.stop_gradient(fa_grad)
    layers.append(x)

    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Align-prop training
def train_alignprop(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                    act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            fa_grad = tf.nn.conv2d(x, W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape)
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) / np.sqrt(
                k * k * last_shape) + b + fa_grad - tf.stop_gradient(fa_grad)
        else:
            fa_grad = tf.nn.conv2d(act(x), W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape)
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) / np.sqrt(
                k * k * last_shape) + b + fa_grad - tf.stop_gradient(fa_grad)
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    fa_grad = tf.nn.conv2d(x, W_lin, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape)
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') / np.sqrt(
        last_shape) + b + fa_grad - tf.stop_gradient(fa_grad)
    layers.append(x)

    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Feedback alignment training with adam, standard parameterization
def train_fa_adam(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                  act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l)) / np.sqrt(k * k * last_shape)
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l)) / np.sqrt(k * k * last_shape)
        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            fa_grad = tf.nn.conv2d(x, W_lin, [1, s, s, 1], p)
            x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, s, s, 1], p) + b + fa_grad - tf.stop_gradient(fa_grad)
        else:
            fa_grad = tf.nn.conv2d(act(x), W_lin, [1, s, s, 1], p)
            x = tf.nn.conv2d(act(tf.stop_gradient(x)), W, [1, s, s, 1], p) + b + fa_grad - tf.stop_gradient(fa_grad)
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10)) / np.sqrt(last_shape)
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10)) / np.sqrt(last_shape)
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    fa_grad = tf.nn.conv2d(x, W_lin, [1, 1, 1, 1], 'VALID')
    x = tf.nn.conv2d(tf.stop_gradient(x), W, [1, 1, 1, 1], 'VALID') + b + fa_grad - tf.stop_gradient(fa_grad)
    layers.append(x)

    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=params)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Align-zero training
def train_alignzero(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                    act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    x_lin = x0
    layers = [x]
    layers_lin = [x_lin]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        if i == 0:
            x_lin = tf.nn.conv2d(x_lin, W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b_lin
        else:
            x_lin = tf.nn.conv2d(act(x_lin), W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b_lin

        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        mean, var = tf.nn.moments(x_lin, [0])
        x_lin = tf.nn.batch_normalization(x_lin, mean, var, None, None, 0.000001)
        layers.append(x)
        layers_lin.append(x_lin)
        last_shape = l

    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
    x_lin = tf.nn.avg_pool(act(x_lin), [1, x_lin.shape[1], x_lin.shape[2], 1],
                           [1, x_lin.shape[1], x_lin.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b
    x_lin = tf.nn.conv2d(x_lin, W_lin, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b_lin
    layers.append(x)

    logits = tf.layers.flatten(x)
    logits_lin = tf.layers.flatten(x_lin)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((tf.stop_gradient(logits) + logits_lin - tf.stop_gradient(logits_lin) - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

    grads_and_vars = opt.compute_gradients(cross_entropy, var_list=params_lin)
    # Replace lin vars with real vars
    new_grads_and_vars = []
    for (g, v), v_real in zip(grads_and_vars, flatten(params)):
        new_grads_and_vars.append((g, v_real))
    optimizer = opt.apply_gradients(new_grads_and_vars)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Align-zero training with adam + standard parameterization
def train_alignzero_adam(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                         act=tf.nn.relu, data_limit=45000, seed=99):
    print(name)
    data = CIFAR()
    x_train = data.train_data[:data_limit] + 0.5
    y_train = data.train_labels[:data_limit] - 0.1
    x_test = data.validation_data[:data_limit] + 0.5
    y_test = data.validation_labels[:data_limit] - 0.1

    # Adjust batch_size
    if batch_size > data_limit:
        batch_size = data_limit
    elif data_limit % batch_size:
        batch_size = np.gcd(batch_size, data_limit)

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    x_lin = x0
    layers = [x]
    layers_lin = [x_lin]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T / np.sqrt(k * k * last_shape)
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
        if i == 0:
            x_lin = tf.nn.conv2d(x_lin, W_lin, [1, s, s, 1], p) + b_lin
        else:
            x_lin = tf.nn.conv2d(act(x_lin), W_lin, [1, s, s, 1], p) + b_lin

        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        mean, var = tf.nn.moments(x_lin, [0])
        x_lin = tf.nn.batch_normalization(x_lin, mean, var, None, None, 0.000001)
        layers.append(x)
        layers_lin.append(x_lin)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
    x_lin = tf.nn.avg_pool(act(x_lin), [1, x_lin.shape[1], x_lin.shape[2], 1],
                           [1, x_lin.shape[1], x_lin.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T / np.sqrt(last_shape)
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') + b
    x_lin = tf.nn.conv2d(x_lin, W_lin, [1, 1, 1, 1], 'VALID') + b_lin
    layers.append(x)

    logits = tf.layers.flatten(x)
    logits_lin = tf.layers.flatten(x_lin)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((tf.stop_gradient(logits) + logits_lin - tf.stop_gradient(logits_lin) - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    opt = tf.train.AdamOptimizer(learning_rate=lr)

    grads_and_vars = opt.compute_gradients(cross_entropy, var_list=params_lin)
    # Replace lin vars with real vars
    new_grads_and_vars = []
    for (g, v), v_real in zip(grads_and_vars, flatten(params)):
        new_grads_and_vars.append((g, v_real))
    optimizer = opt.apply_gradients(new_grads_and_vars)

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


# Last layer training
def train_lastlayer(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                    act=tf.nn.relu):
    print(name)
    data = CIFAR()
    x_train = data.train_data + 0.5
    y_train = data.train_labels - 0.1
    x_test = data.validation_data + 0.5
    y_test = data.validation_labels - 0.1

    np.random.seed(seed)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    x = x0
    layers = [x]
    np.random.seed(seed)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)
        last_shape = l
        layers.append(x)
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    params.append((W, b))
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b
    layers.append(x)
    logits = tf.layers.flatten(x)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=[params[-1]])

    train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=seed)


def train(inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=99):
    start = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(seed)
        for epoch in range(epochs):
            print('Epoch: ' + str(epoch))
            indices = np.random.permutation(x_train.shape[0])
            acc_sum = 0
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :, :, :], labels: y_train[idx, :], lr: lr_val}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                acc_sum += accuracy_value
                step += 1
            print('Train set accuracy: ' + str(acc_sum / (int(x_train.shape[0] / batch_size))))
            if (epoch + 1) % 1 == 0:
                indices = np.random.permutation(x_test.shape[0])
                acc_sum = 0
                for i in range(int(x_test.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_test = {inputs: x_test[idx, :, :, :], labels: y_test[idx, :]}
                    accuracy_value = sess.run(accuracy, feed_dict=feed_dict_test)
                    acc_sum += accuracy_value
                print('Test set accuracy: ' + str(acc_sum / (int(x_test.shape[0] / batch_size))))
    tf.reset_default_graph()
    print('Time ' + str(time.time() - start))


if __name__ == '__main__':
    if part == 1:  # Varying network width experiments
        for lr_val, lr_name in zip([5.0, 2.0, 1.0],
                                   ['50', '20', '10']):
            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7
                for e in [150]:
                    try:
                        train_normal(filters, kernels, strides, paddings,
                                     'cifar_normal_x' + str(scale) + '_lr_' + lr_name,
                                     batch_size=100,
                                     epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignada(filters, kernels, strides, paddings,
                                       'cifar_alignada_x' + str(scale) + '_lr_' + lr_name,
                                       batch_size=100,
                                       epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_dfa(filters, kernels, strides, paddings,
                                  'cifar_dfa_x' + str(scale) + '_lr_' + lr_name,
                                  batch_size=100,
                                  epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignzero(filters, kernels, strides, paddings,
                                        'cifar_alignzero_x' + str(scale) + '_lr_' + lr_name,
                                        batch_size=100,
                                        epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_fa(filters, kernels, strides, paddings,
                                 'cifar_fa_x' + str(scale) + '_lr_' + lr_name,
                                 batch_size=100,
                                 epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_lastlayer(filters, kernels, strides, paddings,
                                        'cifar_lastlayer_x' + str(scale) + '_lr_' + lr_name,
                                        batch_size=100,
                                        epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)
    elif part == 2:  # Varying learning rate experiments
        for lr_val, lr_name in zip([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                                   ['05', '1', '2', '5', '10', '20', '50', '100', '200']):
            for scale in [8, 256]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7
                for e in [150]:
                    try:
                        train_normal(filters, kernels, strides, paddings,
                                     'cifar_normal_x' + str(scale) + '_lr_' + lr_name,
                                     batch_size=100,
                                     epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 256]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_dfa(filters, kernels, strides, paddings,
                                  'cifar_dfa_x' + str(scale) + '_lr_' + lr_name,
                                  batch_size=100,
                                  epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 256]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignada(filters, kernels, strides, paddings,
                                       'cifar_alignada_x' + str(scale) + '_lr_' + lr_name,
                                       batch_size=100,
                                       epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 256]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignzero(filters, kernels, strides, paddings,
                                        'cifar_alignzero_x' + str(scale) + '_lr_' + lr_name,
                                        batch_size=100,
                                        epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 256]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_fa(filters, kernels, strides, paddings,
                                 'cifar_fa_x' + str(scale) + '_lr_' + lr_name,
                                 batch_size=100,
                                 epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 256]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_lastlayer(filters, kernels, strides, paddings,
                                        'cifar_lastlayer_x' + str(scale) + '_lr_' + lr_name,
                                        batch_size=100,
                                        epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

    elif part == 3:  # Standard parameterization experiments
        for lr_val, lr_name in zip([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                                   ['0005', '001', '002', '005', '01', '02', '05', '1', '2', '5']):
            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7
                for e in [150]:
                    try:
                        train_normal_adam(filters, kernels, strides, paddings,
                                          'cifar_normaladam_x' + str(scale) + '_lr_' + lr_name,
                                          batch_size=100,
                                          epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignada_adam(filters, kernels, strides, paddings,
                                            'cifar_alignadaadam_x' + str(scale) + '_lr_' + lr_name,
                                            batch_size=100,
                                            epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignzero_adam(filters, kernels, strides, paddings,
                                             'cifar_alignzeroadam_x' + str(scale) + '_lr_' + lr_name,
                                             batch_size=100,
                                             epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)
    elif part == 4:  # Low data experiments
        for data_lim in [45, 150, 450, 1500, 4500, 15000, 45000]:
            for lr_val, lr_name in zip([5.0, 2.0, 1.0],
                                       ['50', '20', '10']):
                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7
                    for e in [150]:
                        try:
                            train_normal(filters, kernels, strides, paddings,
                                         'cifar_normal_x' + str(scale) + '_lr_' + lr_name + '_d_' + str(data_lim),
                                         batch_size=100,
                                         epochs=e, lr_val=lr_val, data_limit=data_lim)
                        except Exception as e:
                            print(e)

                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_alignada(filters, kernels, strides, paddings,
                                           'cifar_alignada_x' + str(scale) + '_lr_' + lr_name +
                                           '_d_' + str(data_lim),
                                           batch_size=100,
                                           epochs=e, lr_val=lr_val, data_limit=data_lim)
                        except Exception as e:
                            print(e)

                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_alignzero(filters, kernels, strides, paddings,
                                            'cifar_alignzero_x' + str(scale) + '_lr_' + lr_name + '_d_'
                                            + str(data_lim),
                                            batch_size=100,
                                            epochs=e, lr_val=lr_val, data_limit=data_lim)
                        except Exception as e:
                            print(e)

                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_fa(filters, kernels, strides, paddings,
                                     'cifar_fa_x' + str(scale) + '_lr_' + lr_name + '_d_'
                                     + str(data_lim),
                                     batch_size=100,
                                     epochs=e, lr_val=lr_val, data_limit=data_lim)
                        except Exception as e:
                            print(e)

                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_dfa(filters, kernels, strides, paddings,
                                      'cifar_dfa_x' + str(scale) + '_lr_' + lr_name + '_d_'
                                      + str(data_lim),
                                      batch_size=100,
                                      epochs=e, lr_val=lr_val, data_limit=data_lim)
                        except Exception as e:
                            print(e)
    elif part == 5:  # Align-prop experiments
        for lr_val, lr_name in zip([5.0, 2.0, 1.0],
                                   ['50', '20', '10']):
            for scale in [8, 16, 32, 64, 128, 256, 512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [150]:
                    try:
                        train_alignprop(filters, kernels, strides, paddings,
                                        'cifar_alignprop_x' + str(scale) + '_lr_' + lr_name,
                                        batch_size=100,
                                        epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)
    elif part == 6:  # Multiple seeds
        for seed in [100, 101, 102, 103, 104]:
            for lr_val, lr_name in zip([5.0, 2.0, 1.0],
                                       ['50', '20', '10']):
                for scale in [256]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7
                    for e in [150]:
                        try:
                            train_normal(filters, kernels, strides, paddings,
                                         'cifar_normal_x' + str(scale) + '_lr_' + lr_name + '_seed_'
                                         + str(seed),
                                         batch_size=100, seed=seed,
                                         epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)

                for scale in [256]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_alignada(filters, kernels, strides, paddings,
                                           'cifar_alignada_x' + str(scale) + '_lr_' + lr_name
                                           + '_seed_' + str(seed),
                                           batch_size=100, seed=seed,
                                           epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)

                for scale in [256]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_dfa(filters, kernels, strides, paddings,
                                      'cifar_dfa_x' + str(scale) + '_lr_' + lr_name + '_seed_'
                                      + str(seed),
                                      batch_size=100, seed=seed,
                                      epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)

                for scale in [256]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_alignzero(filters, kernels, strides, paddings,
                                            'cifar_alignzero_x' + str(scale) + '_lr_' + lr_name + '_seed_'
                                            + str(seed),
                                            batch_size=100, seed=seed,
                                            epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)

                for scale in [256]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [150]:
                        try:
                            train_fa(filters, kernels, strides, paddings,
                                     'cifar_fa_x' + str(scale) + '_lr_' + lr_name + '_seed_'
                                     + str(seed),
                                     batch_size=100, seed=seed,
                                     epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)
    elif part == 7:  # Training time measurement
        # Warmup run
        filters = [8] * 7
        kernels = [3] * 7
        strides = [1, 1, 1, 2, 1, 2, 1]
        paddings = ['SAME'] * 7
        train_normal(filters, kernels, strides, paddings,
                     'cifar_normal_trial', batch_size=100, epochs=1, lr_val=1.0)

        for seed in [100, 101, 102, 103, 104]:
            for lr_val, lr_name in zip([5.0, 2.0, 1.0],
                                       ['50', '20', '10']):
                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7
                    for e in [1]:
                        try:
                            train_normal(filters, kernels, strides, paddings,
                                         'cifar_normal_x' + str(scale) + '_lr_' + lr_name + '_seed_'
                                         + str(seed),
                                         batch_size=100, seed=seed,
                                         epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)

                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [1]:
                        try:
                            train_alignada(filters, kernels, strides, paddings,
                                           'cifar_alignada_x' + str(scale) + '_lr_' + lr_name
                                           + '_seed_' + str(seed),
                                           batch_size=100, seed=seed,
                                           epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)

                for scale in [8, 16, 32, 64, 128, 256, 512]:
                    filters = [scale] * 7
                    kernels = [3] * 7
                    strides = [1, 1, 1, 2, 1, 2, 1]
                    paddings = ['SAME'] * 7

                    for e in [1]:
                        try:
                            train_alignzero(filters, kernels, strides, paddings,
                                            'cifar_alignzero_x' + str(scale) + '_lr_' + lr_name + '_seed_'
                                            + str(seed),
                                            batch_size=100, seed=seed,
                                            epochs=e, lr_val=lr_val)
                        except Exception as e:
                            print(e)
