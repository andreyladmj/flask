import random
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

from acme.Networks.VGG.fr_utils import list_dir, get_batch_from_peoples, triplet_loss, get_shuffled_training_set, \
    batch_data, triplet_loss2, compute_triplet_loss, preprocess_image
from acme.Networks.VGG.tensorflow_vgg import vgg16
from acme.Networks.VGG.tensorflow_vgg import utils

vgg_dir = 'tensorflow_vgg/'
faces_codes_file = 'faces_codes'
faces_labels_file = 'faces_labels'

import os

import numpy as np
import tensorflow as tf

imw = 96
imh = 96
batch_size = 32

dir = '/home/srivoknovski/dataset/lfw'
#dir = 'E:\dataset\lfw/'
peoples = list_dir(dir, count_of_images=None, count_of_peoples=None)

def train_nn():
    vgg_graph = tf.Graph()
    vgg_session = tf.Session(graph=vgg_graph)

    with vgg_session.as_default():
        with vgg_graph.as_default():
            vgg = vgg16.Vgg16()
            input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
            with tf.name_scope("content_vgg"):
                vgg.build(input_)

    codes, labels = get_codes_and_labels()

    print('labels', labels.shape, 'unique', len(set(labels)))
    print('codes', codes.shape)


    X_train = get_shuffled_training_set(codes, labels)

    print('X_train', X_train.shape)

    tf.reset_default_graph()
    learning_rate = 0.001
    epochs = 400
    iteration = 0

    fcc_graph = tf.Graph()
    fcc_session = tf.Session(graph=fcc_graph)

    def check_predictions():
        print('1:', check(get_im('/Angelina_Jolie/Angelina_Jolie_0004.jpg'), get_im('/Angelina_Jolie/Angelina_Jolie_0007.jpg')), end=', ')
        print('2:', check(get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg'), get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0002.jpg')), end=', ')
        print('3:', check(get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg'), get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0003.jpg')), end=', ')
        print('4*:', check(get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg'), get_im('/Adolfo_Aguilar_Zinser/Adolfo_Aguilar_Zinser_0001.jpg')), end=', ')
        print('5*:', check(get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0002.jpg'), get_im('/Adolfo_Aguilar_Zinser/Adolfo_Aguilar_Zinser_0002.jpg')), end='')
        print('6:', check(get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg'), get_im('/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg')), end=', ')


    def vgg_predict(im1):
        img1 = utils.load_image(im1)
        img1 = img1.reshape((1, 224, 224, 3))
        codes_batch = vgg_session.run(vgg.relu6, feed_dict={input_: img1})
        im1 = codes_batch[0]
        return im1


    def check(im1, im2):
        im1 = vgg_predict(im1)
        im2 = vgg_predict(im2)

        encodec1 = fcc_session.run(anchor, feed_dict={anchor_codes: np.array([im1])})
        encodec2 = fcc_session.run(anchor, feed_dict={anchor_codes: np.array([im2])})
        dist = np.linalg.norm(encodec1 - encodec2)
        return dist


    with fcc_session.as_default():
        with fcc_graph.as_default():

            anchor_codes = tf.placeholder(tf.float32, shape=[None, codes.shape[1]], name='anchor')
            positive_codes = tf.placeholder(tf.float32, shape=[None, codes.shape[1]], name='positive')
            negative_codes = tf.placeholder(tf.float32, shape=[None, codes.shape[1]], name='negative')

            anchor = build_fully_connected_layers(anchor_codes)
            positive = build_fully_connected_layers(positive_codes, reuse=True)
            negative = build_fully_connected_layers(negative_codes, reuse=True)

            #loss, positive_dist, negative_dist = compute_triplet_loss([anchor, positive, negative])
            loss, positive_dist, negative_dist, ttt = triplet_loss([anchor, positive, negative], alpha=10)
            optimizer = tf.train.AdamOptimizer().minimize(loss)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

            fcc_session.run(tf.global_variables_initializer())

            for e in range(epochs):
                for row in batch_data(X_train, batch_size):
                    anc_codes = row[:, 0]
                    pos_codes = row[:, 1]
                    neg_codes = row[:, 2]

                    feed = {anchor_codes: anc_codes, positive_codes: pos_codes, negative_codes: neg_codes}
                    cost, _, p_dist, n_dist, a,p,n = fcc_session.run([loss, optimizer, positive_dist, negative_dist, anchor,positive,negative], feed_dict=feed)

                    dist = [np.linalg.norm(a[0] - p[0]), np.linalg.norm(a[0] - n[0])]

                    print(
                        "Epoch: {}/{}".format(e+1, epochs),
                        "Iteration: {}".format(iteration),
                        "Training loss: {:.5f}".format(cost),
                        "Positive dist: {:.5f}".format(p_dist),
                        "Negative dist: {:.5f}".format(n_dist),
                        "tt: {}".format(dist),
                        end=', '
                    )
                    iteration += 1

                    #if iteration % 500 == 0:
                        #check_predictions()
                    print('')

    saver = tf.train.Saver()
    saver.save(fcc_session, "face_model/model.ckpt")

    fcc_session.close()
    vgg_session.close()


def get_im(path):
    return dir + path


def get_codes_and_labels():
    # read codes and labels from file
    import csv

    with open(faces_labels_file) as f:
        reader = csv.reader(f, delimiter='\n')
        labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    with open(faces_codes_file) as f:
        codes = np.fromfile(f, dtype=np.float32)
        codes = codes.reshape((len(labels), -1))

    return codes, labels


def build_fully_connected_layers(tensor, reuse=False):
    with tf.variable_scope('fc_layers', reuse=reuse) as scope:
        tensor = tf.contrib.layers.fully_connected(tensor, 1024, scope = 'fc1', activation_fn=tf.nn.tanh)
        tensor = tf.contrib.layers.fully_connected(tensor, 128, scope = 'fc2', activation_fn=None)
    return tensor


if __name__ == '__main__':
    train_nn()


