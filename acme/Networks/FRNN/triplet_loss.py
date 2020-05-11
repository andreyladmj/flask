import tensorflow as tf
import numpy as np
#https://github.com/enggen/Deep-Learning-Coursera/blob/master/Convolutional%20Neural%20Networks/Week4/Face%20Recognition/Face%20Recognition%20for%20the%20Happy%20House%20-%20v2.ipynb


def compute_euclidean_distance(x, y):
    d = tf.square(tf.subtract(x, y))
    d = tf.sqrt(tf.reduce_sum(d)) # What about the axis ???
    return d

def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin=0.2):

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

    with tf.name_scope("triplet_loss"):
        d_p_squared = tf.square(compute_euclidean_distance(anchor_feature, positive_feature))
        d_n_squared = tf.square(compute_euclidean_distance(anchor_feature, negative_feature))

        loss = tf.maximum(0., d_p_squared - d_n_squared + margin)
        #loss = d_p_squared - d_n_squared + margin

        return tf.reduce_mean(loss), tf.reduce_mean(d_p_squared), tf.reduce_mean(d_n_squared)



def triplet_loss(y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0), pos_dist, neg_dist

    return loss


def triplet_loss3(y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = (tf.square(tf.subtract(anchor, positive)))
    neg_dist = (tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist, neg_dist), alpha), .0)
    loss = tf.reduce_mean(basic_loss)

    return loss



def triplet_loss_(anchor, positive, negative, margin=.2, extra=False, scope="triplet_loss"):
    r"""Loss for Triplet networks as described in the paper:
    `FaceNet: A Unified Embedding for Face Recognition and Clustering
    <https://arxiv.org/abs/1503.03832>`_
    by Schroff et al.
    Learn embeddings from an anchor point and a similar input (positive) as
    well as a not-similar input (negative).
    Intuitively, a matching pair (anchor, positive) should have a smaller relative distance
    than a non-matching pair (anchor, negative).
    .. math::
        \max(0, m + \Vert a-p\Vert^2 - \Vert a-n\Vert^2)
    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        margin (float): horizon for negative examples
        extra (bool): also return distances for pos and neg.
    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss
