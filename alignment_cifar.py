import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from setup_cifar import CIFAR
import numpy as np
import pickle
import sys
import os

part = 1

if len(sys.argv) > 1:
    part = int(sys.argv[1])


def save(data, name):
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/' + str(name) + '.pkl', 'wb') as file:
        pickle.dump(data, file)


def load(name):
    with open('outputs/' + str(name) + '.pkl', 'rb') as file:
        return pickle.load(file)


def shuffle(a):
    a_shape = a.shape
    a_flat = a.reshape((-1))
    np.random.shuffle(a_flat)
    return a_flat.reshape(a_shape)


# Normal training
def train_normal(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                 act=tf.nn.relu, data_limit=45000):
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

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    x = x0
    layers = [x]
    np.random.seed(99)
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

    train(name, params, inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy)


# Flattens
def flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return [z for y in x for z in flatten(y)]
    else:
        return [x]


# Alignment training
def train_align(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                act=tf.nn.relu):
    print(name)
    data = CIFAR()
    x_train = data.train_data + 0.5
    y_train = data.train_labels - 0.1
    x_test = data.validation_data + 0.5
    y_test = data.validation_labels - 0.1

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 10))
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    params_true = []
    x = x0
    x_lin = x0
    x_true = x0
    layers = [x]
    np.random.seed(99)
    # Define network
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val = np.random.normal(scale=1, size=(l, k * k * last_shape)).T
        W_val = W_val.reshape((k, k, last_shape, l))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_val = np.zeros((l,))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        W_true = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b_true = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

        params.append((W, b))
        params_lin.append((W_lin, b_lin))
        params_true.append((W_true, b_true))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        if i == 0:
            x_lin = tf.nn.conv2d(x_lin, W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b_lin
        else:
            x_lin = tf.nn.conv2d(act(x_lin), W_lin, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b_lin
        if i == 0:
            x_true = tf.nn.conv2d(x_true, W_true, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) \
                     + b_true
        else:
            x_true = tf.nn.conv2d(act(x_true), W_true, [1, s, s, 1], p) / \
                     np.sqrt(k * k * last_shape) + b_true

        # Batch norm
        mean, var = tf.nn.moments(x, [0])
        x = tf.nn.batch_normalization(x, mean, var, None, None, 0.000001)

        mean, var = tf.nn.moments(x_lin, [0])
        x_lin = tf.nn.batch_normalization(x_lin, mean, var, None, None, 0.000001)

        mean, var = tf.nn.moments(x_true, [0])
        x_true = tf.nn.batch_normalization(x_true, mean, var, None, None, 0.000001)
        layers.append(x)
        last_shape = l

    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
    x_lin = tf.nn.avg_pool(act(x_lin), [1, x_lin.shape[1], x_lin.shape[2], 1],
                           [1, x_lin.shape[1], x_lin.shape[2], 1], 'VALID')
    x_true = tf.nn.avg_pool(act(x_true), [1, x_true.shape[1], x_true.shape[2], 1],
                            [1, x_true.shape[1], x_true.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(10, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 10))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((10,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_true = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_true = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    params_true.append((W_true, b_true))
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b
    x_lin = tf.nn.conv2d(x_lin, W_lin, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b_lin
    x_true = tf.nn.conv2d(x_true, W_true, [1, 1, 1, 1], 'VALID') / np.sqrt(last_shape) + b_true
    layers.append(x)

    logits = tf.layers.flatten(x)
    logits_lin = tf.layers.flatten(x_lin)
    logits_true = tf.layers.flatten(x_true)

    predicted_labels = tf.argmax(logits, 1)
    actual_labels = tf.argmax(labels, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, actual_labels), tf.float32))

    cross_entropy = tf.reduce_mean((logits_true + logits_lin - tf.stop_gradient(logits_lin) - labels) ** 2)

    lr = tf.placeholder('float', shape=())
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

    grads_and_vars = opt.compute_gradients(cross_entropy, var_list=params_lin + params_true)
    # Replace lin vars with real vars
    new_grads_and_vars = []
    for (g, v), v_real in zip(grads_and_vars[:len(flatten(params))], flatten(params)):
        new_grads_and_vars.append((g, v_real))
    for g, v in grads_and_vars[len(flatten(params)):]:
        new_grads_and_vars.append((g, v))
    optimizer = opt.apply_gradients(new_grads_and_vars)

    train(name, params, inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy)


def train(name, params, inputs, labels, lr, x_train, x_test, y_train, y_test, epochs, lr_val, batch_size, optimizer,
          cross_entropy, accuracy, seed=99):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(seed)
        param_vals = sess.run(params)
        save(param_vals, name + '_0')
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
            param_vals = sess.run(params)
            save(param_vals, name + '_' + str(epoch + 1))
        save(param_vals, name)
    tf.reset_default_graph()


# Find alignment between two networks
def align(filters, kernels, strides, paddings, name, baseline, initialization, batch_size=100,
          act=tf.nn.relu, iters=100, randomize=False):
    # Load params
    params = load(name)
    params_baseline = load(baseline)
    params_init = load(initialization)

    np.random.seed(99)
    inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
    last_shape = 3

    x = inputs
    noises = [tf.random.normal((batch_size, int(x.shape[1]), int(x.shape[2]), int(x.shape[3])))]
    # Define noises
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        W_val, b_val = params_init[i]
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
        if i == 0:
            x = tf.nn.conv2d(x, W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        else:
            x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) / np.sqrt(k * k * last_shape) + b
        # Batch norm
        noises.append(tf.random.normal((batch_size, int(x.shape[1]), int(x.shape[2]), int(x.shape[3]))))
        last_shape = l

    # Compute alignments
    WDs = []
    WWs = []
    DDs = []
    for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
        # Subtract init value
        if randomize:
            W_val = shuffle(params[i][0]) - params_init[i][0]
        else:
            W_val = params[i][0] - params_init[i][0]
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))

        W_base_val = params_baseline[i][0] - params_init[i][0]
        W_base = tf.Variable(tf.convert_to_tensor(W_base_val, dtype=tf.float32))

        noise_fw = tf.nn.conv2d(noises[i], W, [1, s, s, 1], p)
        backprojection = 1 / int(W.shape[-1]) * \
                         tf.nn.conv2d_transpose(noise_fw, W, (batch_size,) + tuple(noises[i].shape[1:]), [1, s, s, 1],
                                                p)

        noise_fw_base = tf.nn.conv2d(noises[i], W_base, [1, s, s, 1], p)
        backprojection_base = 1 / int(W.shape[-1]) * \
                              tf.nn.conv2d_transpose(noise_fw_base, W_base, (batch_size,) + tuple(noises[i].shape[1:]),
                                                     [1, s, s, 1],
                                                     p)

        backprojection_base = backprojection_base / tf.reduce_mean(tf.abs(backprojection_base))  # Normalize
        backprojection = backprojection / tf.reduce_mean(tf.abs(backprojection))  # Normalize

        weight_data_align = tf.reduce_mean(
            tf.reduce_sum(tf.layers.flatten(backprojection * backprojection_base), axis=1),
            axis=0)
        weight_weight_align = tf.reduce_mean(tf.reduce_sum(tf.layers.flatten(backprojection * backprojection),
                                                           axis=1), axis=0)
        data_data_align = tf.reduce_mean(tf.reduce_sum(tf.layers.flatten(backprojection_base * backprojection_base),
                                                       axis=1), axis=0)
        WDs.append(weight_data_align)
        WWs.append(weight_weight_align)
        DDs.append(data_data_align)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np.random.seed(99)
        WD_sum, WW_sum, DD_sum = None, None, None
        for i in range(iters):
            WD_val, WW_val, DD_val = sess.run([WDs, WWs, DDs])
            update = True
            for v in WD_val:
                if np.isnan(v):
                    update = False
                    break
            for v in WW_val:
                if np.isnan(v):
                    update = False
                    break
            for v in DD_val:
                if np.isnan(v):
                    update = False
                    break
            if update:
                if WD_sum is None:
                    WD_sum = WD_val
                    WW_sum = WW_val
                    DD_sum = DD_val
                else:
                    WD_sum = [s + v for s, v in zip(WD_sum, WD_val)]
                    WW_sum = [s + v for s, v in zip(WW_sum, WW_val)]
                    DD_sum = [s + v for s, v in zip(DD_sum, DD_val)]
        alignment = [WD / np.sqrt(WW * DD) for WD, WW, DD in zip(WD_sum, WW_sum, DD_sum)]
    tf.reset_default_graph()
    return alignment


if __name__ == '__main__':
    if part == 1:  # Train networks to compute alignment
        for scale in [8, 16, 32, 64, 128, 256, 512]:
            filters = [scale] * 7
            kernels = [3] * 7
            strides = [1, 1, 1, 2, 1, 2, 1]
            paddings = ['SAME'] * 7
            for e in [300]:
                try:
                    train_align(filters, kernels, strides, paddings,
                                'cifar_align_x' + str(scale),
                                batch_size=100,
                                epochs=e, lr_val=5.0)
                except Exception as e:
                    print(e)

        for scale in [8, 16, 32, 64, 128, 256, 512]:
            filters = [scale] * 7
            kernels = [3] * 7
            strides = [1, 1, 1, 2, 1, 2, 1]
            paddings = ['SAME'] * 7
            for e in [300]:
                try:
                    train_normal(filters, kernels, strides, paddings,
                                 'cifar_normal_x' + str(scale),
                                 batch_size=100,
                                 epochs=e, lr_val=5.0)
                except Exception as e:
                    print(e)
    elif part == 2:  # Alignment at different network widths
        alignments = []
        for scale in [8, 16, 32, 64, 128, 256, 512]:
            name = 'cifar_normal_x' + str(scale) + '_150'
            baseline = 'cifar_align_x' + str(scale) + '_150'
            initialization = 'cifar_normal_x' + str(scale) + '_0'

            filters = [scale] * 7
            kernels = [3] * 7
            strides = [1, 1, 1, 2, 1, 2, 1]
            paddings = ['SAME'] * 7
            try:
                alignment = align(filters, kernels, strides, paddings, name, baseline, initialization)
                alignments.append(alignment)
            except Exception as e:
                print(e)
        print(alignments)
    elif part == 3:  # Alignment during training
        alignments = []
        for epoch in range(300):
            for scale in [512]:
                name = 'cifar_normal_x' + str(scale) + '_' + str(epoch)
                baseline = 'cifar_align_x' + str(scale) + '_' + str(epoch)
                initialization = 'cifar_normal_x' + str(scale) + '_0'

                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7
                try:
                    alignment = align(filters, kernels, strides, paddings, name, baseline, initialization)
                    alignments.append(alignment)
                except Exception as e:
                    print(e)
        print(alignments)

        alignments = []
        for epoch in range(300):
            for scale in [512]:
                name = 'cifar_normal_x' + str(scale) + '_' + str(epoch)
                baseline = 'cifar_align_x' + str(scale) + '_' + str(epoch)
                initialization = 'cifar_normal_x' + str(scale) + '_0'

                filters = [scale] * 7
                kernels = [3] * 7
                strides = [1, 1, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7
                try:
                    alignment = align(filters, kernels, strides, paddings, name, baseline, initialization,
                                      randomize=True)
                    alignments.append(alignment)
                except Exception as e:
                    print(e)
        print(alignments)
