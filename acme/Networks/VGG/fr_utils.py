import os
import random
import tensorflow as tf
import numpy as np
from PIL import Image


def list_dir(path, count_of_images=10, count_of_peoples=10):
    dict = {}

    for folder in os.listdir(path):
        images = os.listdir(os.path.join(path, folder))

        if len(images) > 10:
            if count_of_peoples:
                count_of_peoples -= 1

            if count_of_images:
                dict[folder] = images[:count_of_images]
            else:
                dict[folder] = images

        if count_of_peoples == 0: break

    return dict


def get_batch_from_peoples(peoples):
    def get_negative_sample(name):
        key = random.choice(list(peoples.keys()))
        while name == key:
            key = random.choice(list(peoples.keys()))
        return os.path.join(key, random.choice(peoples[key]))

    def get_positive_sample(key, curr_image):
        image = random.choice(peoples[key])
        while image == curr_image:
            image = random.choice(peoples[key])
        return os.path.join(key, image)

    for somebody in peoples:
        for image in peoples[somebody]:
            negative = get_negative_sample(somebody)
            positive = get_positive_sample(somebody, image)
            yield os.path.join(somebody, image), positive, negative



def triplet_loss(y_pred, alpha = 0.2, bloss = 10):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ttt = [tf.reduce_sum(anchor), tf.reduce_sum(positive), tf.reduce_sum(negative)]
    #dist = [np.linalg.norm(anchor - positive), np.linalg.norm(anchor - negative)]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    #k = tf.sqrt(tf.abs(tf.reduce_mean(pos_dist)))
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0)

    return loss, pos_dist, neg_dist, ttt


def triplet_loss2(y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = (tf.square(tf.subtract(anchor, positive)))
    neg_dist = (tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist, neg_dist), alpha), .0)
    loss = tf.reduce_mean(basic_loss)

    return loss, pos_dist, neg_dist


def compute_euclidean_distance(x, y):
    d = tf.square(tf.subtract(x, y))
    d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
    return d
def compute_triplet_loss(y_pred, margin=0.2):

    """
    Compute the contrastive loss as in
    L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m
    **Parameters**
     anchor_feature:
     positive_feature:
     negative_feature:
     margin: Triplet margin
    **Returns**
     Return the loss operation
    """
    anchor_feature, positive_feature, negative_feature = y_pred

    with tf.name_scope("triplet_loss"):
        d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
        d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        #loss = d_p_squared - d_n_squared + margin

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)

def get_training_set(codes, labels):
    for i in range(len(labels)):
        label = labels[i]
        keys = [j for j in range(len(labels)) if labels[j] == label and j != i]
        other = [j for j in range(len(labels)) if labels[j] != label and j != i]

        if len(keys):
            a = codes[i]
            p = codes[random.choice(keys)]
            n = codes[random.choice(other)]
            yield a, p, n


def get_shuffled_training_set(codes, labels):
    dataset = []
    for a,p,n in get_training_set(codes, labels):
        dataset.append([a,p,n])
    dataset = np.array(dataset)
    return shuffle(dataset)


def batch_data(data, batch_size):
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        yield data[start:end]


def shuffle(x):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    return x[s]


def preprocess_image(image_path, size=(96,96)):
    im = Image.open(image_path)
    im = im.resize(size)
    im = np.array(im).astype(np.float32)
    im = np.around(im/255.0, decimals=12)
    return im