#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time
import model
import datetime
from tqdm import tqdm


def distort_imgs(data):
    """ data augumentation """
    x1, x2, x3, x4, y = data
    # x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],  # previous without this, hard-dice=83.7
    #                         axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y],
                                                  axis=1, is_random=True)  # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y],
                                                          alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20,
                                                 is_random=True, fill_mode='constant')  # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10,
                                              hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05,
                                              is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y],
                                             zoom_range=[0.9, 1.1], is_random=True,
                                             fill_mode='constant')
    return x1, x2, x3, x4, y


def vis_imgs(X, y, path, show=False):
    """ show one slice """
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
    assert X.ndim == 3
    tl.visualize.save_images(np.asarray([X[:, :, 0, np.newaxis],
                                         X[:, :, 1, np.newaxis], X[:,
                                                                   :, 2, np.newaxis],
                                         X[:, :, 3, np.newaxis], y]), size=(1, 5),
                             image_path=path)
    # if(show):
    # tl.visualize.read_image(path)


def vis_imgs2(X, y_, y, path, show=False):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:, :, np.newaxis]
    assert X.ndim == 3
    tl.visualize.save_images(np.asarray([X[:, :, 0, np.newaxis],
                                         X[:, :, 1, np.newaxis], X[:,
                                                                   :, 2, np.newaxis],
                                         X[:, :, 3, np.newaxis], y_, y]), size=(1, 6),
                             image_path=path)
    # if(show):
    # tl.visualize.read_image(path)


def main(task='all'):
    # Create folder to save trained model and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}".format(task))

    ###======================== LOAD DATA ===================================###
    # by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    import prepare_data_with_valid as dataset
    X_train = dataset.X_train_input
    y_train = dataset.X_train_target[:, :, :, np.newaxis]
    X_val = dataset.X_dev_input
    y_val = dataset.X_dev_target[:, :, :, np.newaxis]
    X_test = dataset.X_test_input
    y_test = dataset.X_test_target[:, :, :, np.newaxis]

    if task == 'all':
        y_train = (y_train > 0).astype(int)
        y_test = (y_test > 0).astype(int)
        y_val = (y_val > 0).astype(int)
    elif task == 'necrotic':
        y_train = (y_train == 1).astype(int)
        y_test = (y_test == 1).astype(int)
        y_val = (y_val == 1).astype(int)
    elif task == 'edema':
        y_train = (y_train == 2).astype(int)
        y_test = (y_test == 2).astype(int)
        y_val = (y_val == 2).astype(int)
    elif task == 'enhance':
        y_train = (y_train == 4).astype(int)
        y_test = (y_test == 4).astype(int)
        y_val = (y_val == 4).astype(int)
    else:
        exit("Unknow task %s" % task)

    ###======================== HYPER-PARAMETERS ============================###
    batch_size = 10
    lr = 0.00001
    # lr_decay = 0.5
    # decay_every = 100
    beta1 = 0.9
    n_epoch = 2
    print_freq_step = 100

    ###======================== SHOW DATA ===================================###
    # show one slice
    X = np.asarray(X_train[80])
    y = np.asarray(y_train[80])
    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    nw, nh, nz = X.shape
    vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augumentation results
    for i in range(batch_size):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:, :, 0, np.newaxis], X[:, :, 1, np.newaxis],
                                                           X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis], y])  # [:,:,np.newaxis]])
        # print(x_flair.shape, x_t1.shape, x_t1ce.shape, x_t2.shape, label.shape) # (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        # print(X_dis.shape, X_dis.min(), X_dis.max()) # (240, 240, 4) -0.380588233471 2.62376139209
        # vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))

    with tf.device('/cpu:0'):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Create folder for tensorboard
        experiment = "lrelu"
        train_log_dir = "logs/{}/".format(task) + experiment + '/train'
        result_log_dir = "logs/{}/".format(task) + experiment + '/res'
        # test_log_dir = "logs/{}/".format(task) + current_time + '/test'

        tl.files.exists_or_mkdir(train_log_dir)
        tl.files.exists_or_mkdir(result_log_dir)
        # train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        # result_writer = tf.summary.FileWriter(result_log_dir, sess.graph)
        # logfile = open("{}/logs.txt".format(train_log_dir), "w")

        # define metric.
        def acc(_logits, y_batch):
            # return np.mean(np.equal(np.argmax(_logits, 1), y_batch))
            return tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(_logits, 1), tf.convert_to_tensor(y_batch, tf.int64)), tf.float32), name='accuracy'
            )
        # train the network
        with tf.device('/gpu:0'):
            t_image = tf.placeholder(
                'float32', [batch_size, nw, nh, nz], name='input_image')
            network = model.u_net(t_image, is_train=True, reuse=False, n_out=1)
            tl.utils.fit(
                network, train_op=tf.optimizers.Adam(learning_rate=0.0001), cost=tl.cost.cross_entropy, X_train=X_train,
                y_train=y_train, acc=acc, batch_size=batch_size, n_epoch=n_epoch, X_val=X_val, y_val=y_val, eval_train=True,
                tensorboard_dir=train_log_dir
            )
        # test
        tl.utils.test(network, acc, X_test, y_test,
                      batch_size=None, cost=tl.cost.cross_entropy)
        # evaluation
        _logits = tl.utils.predict(network, X_test)
        y_pred = np.argmax(_logits, 1)
        tl.utils.evaluation(y_test, y_pred, n_classes=10)
        # save network weights
        network.save_weights('model.h5')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='all',
                        help='all, necrotic, edema, enhance')

    args = parser.parse_args()

    main(args.task)
