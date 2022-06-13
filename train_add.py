import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
import time
import sys

part = 1

if len(sys.argv) > 1:
    part = int(sys.argv[1])


def gen_data(t1, t2, T=100):
    np.random.seed(99)
    x = np.random.binomial(1, 0.5, size=(400, T))
    roll1 = np.roll(x, t1, axis=1)
    roll1[:t1] = 0
    roll2 = np.roll(x, t2, axis=1)
    roll2[:t2] = 0
    y = 0.5 + 0.5 * roll1 - 0.25 * roll2
    return x[:300, :], y[:300, :], x[300:, :], y[300:, :]


# Normal training
def train_normal(n_hidden, t1, t2, T=100, lr_val=0.001, batch_size=50, epochs=200):
    x_train, y_train, x_test, y_test = gen_data(t1, t2, T)
    print('Random chance test loss: ' + str(np.sum(np.mean((y_test - 0.5) ** 2, axis=0))))
    print('Random chance train loss: ' + str(np.sum(np.mean((y_train - 0.5) ** 2, axis=0))))
    print('N hidden: ' + str(n_hidden))

    np.random.seed(99)
    inputs = tf.placeholder('float', shape=(None, T))
    labels = tf.placeholder('float', shape=(None, T))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ih = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(n_hidden, n_hidden))
    W_hh = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((n_hidden,))
    b_h = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ho = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros(1)
    b_o = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    init_state_val = np.zeros(shape=(batch_size, n_hidden))
    init_state = tf.Variable(tf.convert_to_tensor(init_state_val, dtype=tf.float32))

    state = init_state
    loss = 0
    for i in range(T):
        state = tf.nn.relu(tf.tensordot(inputs[:, i], W_ih, axes=0)
                           + tf.tensordot(state, W_hh, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_h)
        output = tf.tensordot(state, W_ho, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_o
        loss = loss + tf.reduce_mean((output - labels[:, i]) ** 2, axis=0)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,
                                                                             var_list=[W_ih, W_hh, b_h, W_ho, b_o,
                                                                                       init_state])

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        best_train_loss = np.inf
        best_test_loss = np.inf
        for epoch in range(epochs):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :], lr: lr_val}
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict_train)
                step += 1

            if (epoch + 1) % 1 == 0:
                indices = np.random.permutation(x_test.shape[0])
                loss_sum = 0
                for i in range(int(x_test.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_test = {inputs: x_test[idx, :], labels: y_test[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_test)
                    loss_sum += loss_val
                print('Test set loss: ' + str(loss_sum / (int(x_test.shape[0] / batch_size))))
                if loss_sum / (int(x_test.shape[0] / batch_size)) < best_test_loss:
                    best_test_loss = loss_sum / (int(x_test.shape[0] / batch_size))

                indices = np.random.permutation(x_train.shape[0])
                loss_sum = 0
                for i in range(int(x_train.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_train)
                    loss_sum += loss_val
                print('Train set loss: ' + str(loss_sum / (int(x_train.shape[0] / batch_size))))
                if loss_sum / (int(x_train.shape[0] / batch_size)) < best_train_loss:
                    best_train_loss = loss_sum / (int(x_train.shape[0] / batch_size))
    return best_train_loss, best_test_loss


# Readout only training
def train_readout(n_hidden, t1, t2, T=100, lr_val=0.001, batch_size=50, epochs=200):
    x_train, y_train, x_test, y_test = gen_data(t1, t2, T)
    print('Random chance test loss: ' + str(np.sum(np.mean((y_test - 0.5) ** 2, axis=0))))
    print('Random chance train loss: ' + str(np.sum(np.mean((y_train - 0.5) ** 2, axis=0))))
    print('N hidden: ' + str(n_hidden))

    np.random.seed(99)
    inputs = tf.placeholder('float', shape=(None, T))
    labels = tf.placeholder('float', shape=(None, T))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ih = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(n_hidden, n_hidden))
    W_hh = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((n_hidden,))
    b_h = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ho = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros(1)
    b_o = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    init_state_val = np.zeros(shape=(batch_size, n_hidden))
    init_state = tf.Variable(tf.convert_to_tensor(init_state_val, dtype=tf.float32))

    state = init_state
    loss = 0
    for i in range(T):
        state = tf.nn.relu(tf.tensordot(inputs[:, i], W_ih, axes=0)
                           + tf.tensordot(state, W_hh, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_h)
        output = tf.tensordot(state, W_ho, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_o
        loss = loss + tf.reduce_mean((output - labels[:, i]) ** 2, axis=0)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,
                                                                             var_list=[W_ho, b_o,
                                                                                       init_state])

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        best_train_loss = np.inf
        best_test_loss = np.inf
        for epoch in range(epochs):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :], lr: lr_val}
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict_train)
                step += 1

            if (epoch + 1) % 1 == 0:
                indices = np.random.permutation(x_test.shape[0])
                loss_sum = 0
                for i in range(int(x_test.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_test = {inputs: x_test[idx, :], labels: y_test[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_test)
                    loss_sum += loss_val
                print('Test set loss: ' + str(loss_sum / (int(x_test.shape[0] / batch_size))))
                if loss_sum / (int(x_test.shape[0] / batch_size)) < best_test_loss:
                    best_test_loss = loss_sum / (int(x_test.shape[0] / batch_size))

                indices = np.random.permutation(x_train.shape[0])
                loss_sum = 0
                for i in range(int(x_train.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_train)
                    loss_sum += loss_val
                print('Train set loss: ' + str(loss_sum / (int(x_train.shape[0] / batch_size))))
                if loss_sum / (int(x_train.shape[0] / batch_size)) < best_train_loss:
                    best_train_loss = loss_sum / (int(x_train.shape[0] / batch_size))
    return best_train_loss, best_test_loss


# Semialignzero training
def train_alignada(n_hidden, t1, t2, T=100, lr_val=0.001, batch_size=50, epochs=200):
    x_train, y_train, x_test, y_test = gen_data(t1, t2, T)
    print('Random chance test loss: ' + str(np.sum(np.mean((y_test - 0.5) ** 2, axis=0))))
    print('Random chance train loss: ' + str(np.sum(np.mean((y_train - 0.5) ** 2, axis=0))))
    print('N hidden: ' + str(n_hidden))

    np.random.seed(99)
    inputs = tf.placeholder('float', shape=(None, T))
    labels = tf.placeholder('float', shape=(None, T))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ih = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    W_ih_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(n_hidden, n_hidden))
    W_hh = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    W_hh_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((n_hidden,))
    b_h = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    b_h_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ho = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    W_ho_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros(1)
    b_o = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    b_o_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    init_state_val = np.zeros(shape=(batch_size, n_hidden))
    init_state = tf.Variable(tf.convert_to_tensor(init_state_val, dtype=tf.float32))

    state = init_state
    state_lin = tf.stop_gradient(init_state)
    loss = 0
    for i in range(T):
        state_lin = tf.nn.relu(tf.tensordot(inputs[:, i], W_ih_lin, axes=0)
                               + tf.tensordot(state_lin + state - tf.stop_gradient(state), W_hh_lin,
                                              axes=[[1], [0]]) / np.sqrt(n_hidden) + b_h_lin)
        state = tf.nn.relu(tf.tensordot(inputs[:, i], W_ih, axes=0)
                           + tf.tensordot(tf.stop_gradient(state), W_hh, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_h)
        output = tf.tensordot(tf.stop_gradient(state), W_ho, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_o
        output_lin = tf.tensordot(state_lin, W_ho_lin, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_o_lin
        loss = loss + tf.reduce_mean((output + output_lin - tf.stop_gradient(output_lin) - labels[:, i]) ** 2, axis=0)

    lr = tf.placeholder('float', shape=())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss,
                                                                             var_list=[W_ih, W_hh, b_h, W_ho, b_o,
                                                                                       init_state])

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        best_train_loss = np.inf
        best_test_loss = np.inf
        for epoch in range(epochs):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :], lr: lr_val}
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict_train)
                step += 1

            if (epoch + 1) % 1 == 0:
                indices = np.random.permutation(x_test.shape[0])
                loss_sum = 0
                for i in range(int(x_test.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_test = {inputs: x_test[idx, :], labels: y_test[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_test)
                    loss_sum += loss_val
                print('Test set loss: ' + str(loss_sum / (int(x_test.shape[0] / batch_size))))
                if loss_sum / (int(x_test.shape[0] / batch_size)) < best_test_loss:
                    best_test_loss = loss_sum / (int(x_test.shape[0] / batch_size))

                indices = np.random.permutation(x_train.shape[0])
                loss_sum = 0
                for i in range(int(x_train.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_train)
                    loss_sum += loss_val
                print('Train set loss: ' + str(loss_sum / (int(x_train.shape[0] / batch_size))))
                if loss_sum / (int(x_train.shape[0] / batch_size)) < best_train_loss:
                    best_train_loss = loss_sum / (int(x_train.shape[0] / batch_size))
    return best_train_loss, best_test_loss


# Align-zero training
def train_alignzero(n_hidden, t1, t2, T=100, lr_val=0.001, batch_size=50, epochs=200):
    x_train, y_train, x_test, y_test = gen_data(t1, t2, T)
    print('Random chance test loss: ' + str(np.sum(np.mean((y_test - 0.5) ** 2, axis=0))))
    print('Random chance train loss: ' + str(np.sum(np.mean((y_train - 0.5) ** 2, axis=0))))
    print('N hidden: ' + str(n_hidden))

    np.random.seed(99)
    inputs = tf.placeholder('float', shape=(None, T))
    labels = tf.placeholder('float', shape=(None, T))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ih = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    W_ih_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(n_hidden, n_hidden))
    W_hh = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    W_hh_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((n_hidden,))
    b_h = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    b_h_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=n_hidden)
    W_ho = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    W_ho_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros(1)
    b_o = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    b_o_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    init_state_val = np.zeros(shape=(batch_size, n_hidden))
    init_state = tf.Variable(tf.convert_to_tensor(init_state_val, dtype=tf.float32))

    state = init_state
    state_lin = init_state
    loss = 0
    for i in range(T):
        state_lin = tf.nn.relu(tf.tensordot(inputs[:, i], W_ih_lin, axes=0)
                               + tf.tensordot(state_lin, W_hh_lin,
                                              axes=[[1], [0]]) / np.sqrt(n_hidden) + b_h_lin)
        state = tf.nn.relu(tf.tensordot(inputs[:, i], W_ih, axes=0)
                           + tf.tensordot(state, W_hh, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_h)
        output = tf.tensordot(state, W_ho, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_o
        output_lin = tf.tensordot(state_lin, W_ho_lin, axes=[[1], [0]]) / np.sqrt(n_hidden) + b_o_lin
        loss = loss + tf.reduce_mean(
            (tf.stop_gradient(output) + output_lin - tf.stop_gradient(output_lin) - labels[:, i]) ** 2, axis=0)

    lr = tf.placeholder('float', shape=())
    params = [W_ih, W_hh, b_h, W_ho, b_o, init_state]
    params_lin = [W_ih_lin, W_hh_lin, b_h_lin, W_ho_lin, b_o_lin, init_state]
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)

    grads_and_vars = opt.compute_gradients(loss, var_list=params_lin)
    # Replace lin vars with real vars
    new_grads_and_vars = []
    for (g, v), v_real in zip(grads_and_vars, params):
        new_grads_and_vars.append((g, v_real))
    optimizer = opt.apply_gradients(new_grads_and_vars)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        best_train_loss = np.inf
        best_test_loss = np.inf
        for epoch in range(epochs):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :], lr: lr_val}
                _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict_train)
                step += 1

            if (epoch + 1) % 1 == 0:
                indices = np.random.permutation(x_test.shape[0])
                loss_sum = 0
                for i in range(int(x_test.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_test = {inputs: x_test[idx, :], labels: y_test[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_test)
                    loss_sum += loss_val
                print('Test set loss: ' + str(loss_sum / (int(x_test.shape[0] / batch_size))))
                if loss_sum / (int(x_test.shape[0] / batch_size)) < best_test_loss:
                    best_test_loss = loss_sum / (int(x_test.shape[0] / batch_size))

                indices = np.random.permutation(x_train.shape[0])
                loss_sum = 0
                for i in range(int(x_train.shape[0] / batch_size)):
                    idx = indices[i * batch_size: (i + 1) * batch_size]
                    feed_dict_train = {inputs: x_train[idx, :], labels: y_train[idx, :]}
                    loss_val = sess.run(loss, feed_dict=feed_dict_train)
                    loss_sum += loss_val
                print('Train set loss: ' + str(loss_sum / (int(x_train.shape[0] / batch_size))))
                if loss_sum / (int(x_train.shape[0] / batch_size)) < best_train_loss:
                    best_train_loss = loss_sum / (int(x_train.shape[0] / batch_size))
    return best_train_loss, best_test_loss


if __name__ == '__main__':
    if part == 1:  # All Add task experiments
        losses = []
        for n_hidden in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for e in [200]:
                try:
                    loss = train_normal(n_hidden, 2, 5, epochs=e)
                    losses.append(loss)
                except Exception as e:
                    print(e)
            print(losses)

        losses = []
        for n_hidden in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for e in [200]:
                try:
                    loss = train_alignada(n_hidden, 2, 5, epochs=e)
                    losses.append(loss)
                except Exception as e:
                    print(e)
            print(losses)

        losses = []
        for n_hidden in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for e in [200]:
                try:
                    loss = train_alignzero(n_hidden, 2, 5, epochs=e)
                    losses.append(loss)
                except Exception as e:
                    print(e)
            print(losses)

        losses = []
        for n_hidden in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for e in [200]:
                try:
                    loss = train_readout(n_hidden, 2, 5, epochs=e)
                    losses.append(loss)
                except Exception as e:
                    print(e)
            print(losses)
