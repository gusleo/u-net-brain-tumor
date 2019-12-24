#! /usr/bin/python
# -*- coding: utf8 -*-

import model
import os
import pickle
import nibabel as nib

import numpy as np
import tensorflow as tf
import tensorlayer as tl
import argparse


def vis_imgs_with_pred(X, y_, y, path):
    """ show one slice with target """
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
    if y_.ndim == 2:
        y_ = y_[:, :, np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:, :, 0, np.newaxis],
                                   X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis],
                                   X[:, :, 3, np.newaxis], y_, y]), size=(1, 6),
                       image_path=path)


def load_image():
    X_dev_input = []
    X_dev_target = []
    with open('data/train_dev_all/mean_std_dict.pickle', 'rb') as file:
        data_types_mean_std_dict = pickle.load(file)

    filename = 'Brats17_TCIA_175_1'
    LGG_data_path = 'data/Brats17TrainingData/LGG/'
    all_3d_data = []
    for img_type in ['flair', 't1', 't1ce', 't2']:
        img_path = os.path.join(LGG_data_path, filename,
                                filename + '_' + img_type + '.nii.gz')
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[img_type]['mean']
               ) / data_types_mean_std_dict[img_type]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

        seg_path = os.path.join(LGG_data_path, filename,
                                filename + '_seg.nii.gz')
        seg_img = nib.load(seg_path).get_data()
        seg_img = np.transpose(seg_img, (1, 0, 2))
    for piece in range(all_3d_data[0].shape[2]):
        combined_array = np.stack(
            (all_3d_data[0][:, :, piece], all_3d_data[1][:, :, piece],
             all_3d_data[2][:, :, piece], all_3d_data[3][:, :, piece]),
            axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))  # .tolist()
        combined_array.astype(np.float32)
        X_dev_input.append(combined_array)

        seg_2d = seg_img[:, :, piece]
        seg_2d.astype(int)
        X_dev_target.append(seg_2d)
    print("finished {}".format(filename))

    X_dev_input = np.asarray(X_dev_input, dtype=np.float32)
    X_dev_target = np.asarray(X_dev_target)
    return X_dev_input, X_dev_target


def main():
    ###======================== HYPER-PARAMETERS ============================###
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='all',
                        help='all, necrotic, edema, enhance')

    args = parser.parse_args()
    task = args.task

    # Create folder to save trained model and result images
    save_dir = "checkpoint"
    experiment = "lreluwithbias"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/{}/{}".format(task, experiment))

    ###======================== LOAD DATA ===================================###
    # by importing this, you can load a training set and a validation set.
    # you will get X_train_input, X_train_target, X_dev_input and X_dev_target
    # there are 4 labels in targets:
    # Label 0: background
    # Label 1: necrotic and non-enhancing tumor
    # Label 2: edema
    # Label 4: enhancing tumor
    X_test, y_test = load_image()
    X_test = X_test[91][np.newaxis, :, :, :]
    y_test = y_test[:, :, :, np.newaxis][91][np.newaxis, :, :, :]

    if task == 'all':
        y_test = (y_test > 0).astype(int)
    elif task == 'necrotic':
        y_test = (y_test == 1).astype(int)
    elif task == 'edema':
        y_test = (y_test == 2).astype(int)
    elif task == 'enhance':
        y_test = (y_test == 4).astype(int)
    else:
        exit("Unknown task %s" % task)

    ###======================== SHOW DATA ===================================###
    if not os.path.exists('outputs/{}/{}'.format(task, experiment)):
        os.makedirs('outputs/{}/{}'.format(task, experiment))
    X = np.asarray(X_test[0])
    y = np.asarray(y_test[0])
    vis_imgs_with_pred(
        X, y, y, "outputs/{}/{}/run_input.png".format(task, experiment))

    ###======================== TRAIN  ===================================###
    nw, nh, nz = X.shape
    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/cpu:0'):  # <- remove it if you train on CPU or other GPU
            ###======================== DEFINE MODEL =======================###
            # nz is 4 as we input all Flair, T1, T1c and T2.
            t_image = tf.placeholder(
                'float32', [1, nw, nh, nz], name='input_image')
            # labels are either 0 or 1
            t_seg = tf.placeholder(
                'float32', [1, nw, nh, 1], name='target_segment')
            # test inference
            #net_test = model.u_net(t_image, is_train=False, reuse=False, n_out=1)
            net_test = model.u_net_bn_relu(
                t_image, is_train=False, reuse=False, n_out=1)

            ###======================== DEFINE LOSS =========================###

            # test losses
            test_out_seg = net_test.outputs
            # , 'jaccard', epsilon=1e-5)
            test_dice_loss = 1 - \
                tl.cost.dice_coe(test_out_seg, t_seg, axis=(0, 1, 2, 3))
            test_iou_loss = tl.cost.iou_coe(
                test_out_seg, t_seg, axis=(0, 1, 2, 3))
            test_dice_hard = tl.cost.dice_hard_coe(
                test_out_seg, t_seg, axis=(0, 1, 2, 3))

            # ----
            test_correct_prediction = tf.equal(
                tf.argmax(test_out_seg, 1), tf.argmax(t_seg, 1))
            test_acc = tf.reduce_mean(
                tf.cast(test_correct_prediction, tf.float32))

        ###======================== LOAD MODEL ==============================###
        tl.layers.initialize_global_variables(sess)
        # load existing model if possible
        tl.files.load_and_assign_npz(
            sess=sess, name=save_dir + '/u_net_{}_{}.npz'.format(task, experiment), network=net_test)

        ###======================== EVALUATION ==========================###
        total_dice, total_iou, total_dice_hard, total_acc, n_batch = 0, 0, 0, 0, 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test,
                                            batch_size=1, shuffle=True):
            b_images, b_labels = batch
            _dice, _iou, _diceh, _acc, out = sess.run([test_dice_loss,
                                                       test_iou_loss, test_dice_hard, test_acc, net_test.outputs],
                                                      {t_image: b_images, t_seg: b_labels})
            total_dice += _dice
            total_iou += _iou
            total_dice_hard += _diceh
            total_acc += _acc
            n_batch += 1

            vis_imgs_with_pred(b_images[0], b_labels[0], out[0],
                               "outputs/{}/{}/test_{}.png".format(task, experiment, 0))

        print(" **" + " " * 17 + "test accuracy: %f test 1-dice: %f hard-dice: %f iou: %f (2d no distortion)" %
              (total_acc/n_batch, total_dice / n_batch, total_dice_hard / n_batch, total_iou / n_batch))
        print(" task: {}".format(task))
        # save a prediction of test set


if __name__ == "__main__":
    main()
