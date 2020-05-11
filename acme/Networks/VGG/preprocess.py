import random
from urllib.request import urlretrieve
from os.path import isfile, isdir

import cv2
from tqdm import tqdm

from acme.Networks.VGG.align_dlib import AlignDlib
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
align_dlib = AlignDlib('/home/srivoknovski/Python/flask/acme/Networks/FaceNet/shape_predictor_68_face_landmarks.dat')

def preproccess_images():
    check_vgg()
    check_codes()

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

    def vgg_predict(im1):
        img1 = utils.load_image(im1)
        img1 = img1.reshape((1, 224, 224, 3))
        codes_batch = vgg_session.run(vgg.relu6, feed_dict={input_: img1})
        im1 = codes_batch[0]
        return im1

def get_im(path):
    return dir + path

from PIL import Image
import matplotlib.pyplot as plt
def get_people_codes(peoples):
    tmp_var=0
    codes = []
    labels = []

    with tf.Session() as sess:
        vgg = vgg16.Vgg16()
        input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.name_scope("content_vgg"):
            vgg.build(input_)

        for somebody_name in peoples:
            # tmp_var+=1
            # if tmp_var == 10: break

            print("Starting {} images".format(somebody_name))
            class_path = os.path.join(dir, somebody_name)
            files = os.listdir(class_path)
            for ii, file in enumerate(files, 1):

                file_path = os.path.join(class_path, file)
                align_img = align_process_image(file_path, crop_dim=224)

                if align_img is None: continue

                img = align_img / 255.0
                #img = utils.load_image(file_path)

                feed_dict = {input_: [img]}
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                codes.append(codes_batch[0])
                labels.append(somebody_name)
                print(ii, 'of', len(files))

    return codes, labels

def save_codes_and_labels(codes, labels):
    # write codes to file
    with open(faces_codes_file, 'w') as f:
        codes.tofile(f)

    # write labels to file
    import csv
    with open(faces_labels_file, 'w') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(labels)


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


def check_codes():
    if not isfile(faces_codes_file):
        codes, labels = get_people_codes(peoples)
        codes = np.array(codes)
        print('codes', codes.shape)
        print('labels', len(labels))
        save_codes_and_labels(codes, labels)


def align_process_image(filename, crop_dim=180):
    image = None
    aligned_image = None

    image = align_buffer_image(filename)

    if image is not None:
        aligned_image = align_image(image, crop_dim)
    else:
        raise IOError('Error buffering image: {}'.format(filename))

    return aligned_image


def align_buffer_image(filename):
    image = cv2.imread(filename, )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def align_image(image, crop_dim):
    bb = align_dlib.getLargestFaceBoundingBox(image)
    aligned = align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if aligned is not None:
        aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    return aligned


def check_vgg():
    vgg_dir = 'tensorflow_vgg/'
    # Make sure vgg exists
    if not isdir(vgg_dir):
        raise Exception("VGG directory doesn't exist!")

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(vgg_dir + "vgg16.npy"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='VGG16 Parameters') as pbar:
            urlretrieve(
                'https://s3.amazonaws.com/content.udacity-data.com/nd101/vgg16.npy',
                vgg_dir + 'vgg16.npy',
                pbar.hook)
    else:
        print("Parameter file already exists!")

if __name__ == '__main__':
    preproccess_images()


