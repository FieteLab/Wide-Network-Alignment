import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch
import numpy as np
import time
import sys

part = 1

if len(sys.argv) > 1:
    part = int(sys.argv[1])


# Normal training
def train_normal(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                 act=tf.nn.relu):
    print(name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = ImageFolder('../train/', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    test_data = ImageFolder('../val/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 1000))
    inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
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
    W_val = np.random.normal(scale=1, size=(1000, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 1000))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((1000,))
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

    train(params, inputs, labels, lr, train_loader, test_loader, epochs, lr_val, optimizer,
          cross_entropy, accuracy)


# Flattens
def flatten(x):
    if isinstance(x, list) or isinstance(x, tuple):
        return [z for y in x for z in flatten(y)]
    else:
        return [x]


# Align-zero training
def train_alignzero(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                    act=tf.nn.relu):
    print(name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = ImageFolder('../train/', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    test_data = ImageFolder('../val/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 1000))
    inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    x_lin = x0
    layers = [x]
    layers_lin = [x_lin]
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
    W_val = np.random.normal(scale=1, size=(1000, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 1000))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((1000,))
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

    train(params, inputs, labels, lr, train_loader, test_loader, epochs, lr_val, optimizer,
          cross_entropy, accuracy)


# Align-ada training
def train_alignada(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
                   act=tf.nn.relu):
    print(name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = ImageFolder('../train/', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    test_data = ImageFolder('../val/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 1000))
    inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
    x = x0
    x_lin = x0
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
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
    x_lin = tf.nn.avg_pool(act(x_lin), [1, x_lin.shape[1], x_lin.shape[2], 1],
                           [1, x_lin.shape[1], x_lin.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(1000, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 1000))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((1000,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))
    W_lin = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_lin = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    params.append((W, b))
    params_lin.append((W_lin, b_lin))
    x_lin = tf.nn.conv2d(x_lin + x - tf.stop_gradient(x), W_lin, [1, 1, 1, 1], 'VALID') / \
            np.sqrt(last_shape) + b_lin
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

    train(params, inputs, labels, lr, train_loader, test_loader, epochs, lr_val, optimizer,
          cross_entropy, accuracy)


# Direct feedback alignment training
def train_dfa(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
              act=tf.nn.relu):
    print(name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = ImageFolder('../train/', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    test_data = ImageFolder('../val/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 1000))
    inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
    last_shape = 3
    cur_size = 224
    n_classes = 1000

    x0 = inputs

    params = []
    x = x0
    out_grad = tf.zeros_like(labels)
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
        x_pool = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')
        proj = tf.layers.flatten(tf.nn.conv2d(x_pool, B, [1, cur_size, cur_size, 1], 'VALID'))
        out_grad = out_grad + proj - tf.stop_gradient(proj)
        layers.append(x)
        last_shape = l
    # Global average pooling
    x = tf.nn.avg_pool(act(x), [1, x.shape[1], x.shape[2], 1], [1, x.shape[1], x.shape[2], 1], 'VALID')

    # Final layer
    W_val = np.random.normal(scale=1, size=(1000, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 1000))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((1000,))
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

    train(params, inputs, labels, lr, train_loader, test_loader, epochs, lr_val, optimizer,
          cross_entropy, accuracy)


# Feedback alignment training
def train_fa(filters, kernels, strides, paddings, name, lr_val=2.0, batch_size=100, epochs=200,
             act=tf.nn.relu):
    print(name)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_data = ImageFolder('../train/', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    test_data = ImageFolder('../val/', transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    np.random.seed(99)
    labels = tf.placeholder('float', shape=(None, 1000))
    inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
    last_shape = 3

    x0 = inputs

    params = []
    params_lin = []
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
    W_val = np.random.normal(scale=1, size=(1000, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 1000))
    W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
    b_val = np.zeros((1000,))
    b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

    W_val = np.random.normal(scale=1, size=(1000, last_shape)).T
    W_val = W_val.reshape((1, 1, last_shape, 1000))
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

    train(params, inputs, labels, lr, train_loader, test_loader, epochs, lr_val, optimizer,
          cross_entropy, accuracy)


def train(params, inputs, labels, lr, train_loader, test_loader, epochs, lr_val, optimizer,
          cross_entropy, accuracy):
    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        param_vals = sess.run(params)
        for epoch in range(epochs):
            print('Epoch: ' + str(epoch))
            acc_sum = 0
            t_count = 0
            for x_batch, y_batch in train_loader:
                one_hot = np.zeros((y_batch.shape[0], 1000))
                one_hot[np.arange(y_batch.shape[0]), y_batch] = 1

                feed_dict_train = {inputs: np.transpose(x_batch, (0, 2, 3, 1)), labels: one_hot, lr: lr_val}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                if t_count % 100 == 0:
                    print('Train iteration: ' + str(t_count))
                    print('Current loss: ' + str(cross_entropy_value))
                    print('Current accuracy: ' + str(accuracy_value))
                acc_sum += accuracy_value
                step += 1
                t_count += 1
            print('Train set accuracy: ' + str(acc_sum / t_count))
            if (epoch + 1) % 1 == 0:
                acc_sum = 0
                t_count = 0
                for x_batch, y_batch in test_loader:
                    one_hot = np.zeros((y_batch.shape[0], 1000))
                    one_hot[np.arange(y_batch.shape[0]), y_batch] = 1

                    feed_dict_test = {inputs: np.transpose(x_batch, (0, 2, 3, 1)), labels: one_hot}
                    accuracy_value = sess.run(accuracy, feed_dict=feed_dict_test)
                    if t_count % 100 == 0:
                        print('Test iteration: ' + str(t_count))
                        print('Current accuracy: ' + str(accuracy_value))
                    acc_sum += accuracy_value
                    t_count += 1
                print('Test set accuracy: ' + str(acc_sum / t_count))
    tf.reset_default_graph()
    print('Time ' + str(time.time() - start))


if __name__ == '__main__':
    if part == 1:  # All ImageNet experiments
        for lr_val, lr_name in zip([5.0],
                                   ['50']):
            for scale in [512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [4, 2, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7
                for e in [300]:
                    try:
                        train_normal(filters, kernels, strides, paddings,
                                     'imagenet_normal_x' + str(scale) + '_lr_' + lr_name,
                                     batch_size=100,
                                     epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [4, 2, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [300]:
                    try:
                        train_alignada(filters, kernels, strides, paddings,
                                       'imagenet_alignada_x' + str(scale) + '_lr_' + lr_name,
                                       batch_size=100,
                                       epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [4, 2, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [300]:
                    try:
                        train_dfa(filters, kernels, strides, paddings,
                                  'imagenet_dfa_x' + str(scale) + '_lr_' + lr_name,
                                  batch_size=100,
                                  epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [4, 2, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [300]:
                    try:
                        train_fa(filters, kernels, strides, paddings,
                                 'imagenet_fa_x' + str(scale) + '_lr_' + lr_name,
                                 batch_size=100,
                                 epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)

            for scale in [512]:
                filters = [scale] * 7
                kernels = [3] * 7
                strides = [4, 2, 1, 2, 1, 2, 1]
                paddings = ['SAME'] * 7

                for e in [300]:
                    try:
                        train_alignzero(filters, kernels, strides, paddings,
                                        'imagenet_alignzero_x' + str(scale) + '_lr_' + lr_name,
                                        batch_size=100,
                                        epochs=e, lr_val=lr_val)
                    except Exception as e:
                        print(e)
