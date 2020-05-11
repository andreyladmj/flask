import tensorflow as tf
import csv
import numpy as np

from acme.Networks.VGG.fr_utils import get_shuffled_training_set, triplet_loss, batch_data

faces_codes_file = 'faces_codes'
faces_labels_file = 'faces_labels'

with open(faces_labels_file) as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
with open(faces_codes_file) as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))

print('labels', labels.shape, len(set(labels)))
print('codes', codes.shape)


X_train = get_shuffled_training_set(codes, labels)

print('X_train', X_train.shape)

tf.reset_default_graph()
learning_rate = 0.001

def build_fully_connected_layers(tensor, reuse=False):
    with tf.variable_scope('fc_layers', reuse=reuse) as scope:
        # tensor = tf.contrib.layers.flatten(tensor)
        #tensor = tf.contrib.layers.fully_connected(tensor, 512, scope = 'fc1')
        tensor = tf.contrib.layers.fully_connected(tensor, 4096, activation_fn=None, scope = 'fc2')
    return tensor

print('codes.shape[1]', codes.shape[1])

anchor_codes = tf.placeholder(tf.float32, shape=[None, codes.shape[1]], name='anchor')
positive_codes = tf.placeholder(tf.float32, shape=[None, codes.shape[1]], name='positive')
negative_codes = tf.placeholder(tf.float32, shape=[None, codes.shape[1]], name='negative')

#anchor = build_fully_connected_layers(anchor_codes)
#positive = build_fully_connected_layers(anchor_codes, reuse=True)
#negative = build_fully_connected_layers(anchor_codes, reuse=True)


anchor = tf.contrib.layers.fully_connected(anchor_codes, 1024, activation_fn=None, scope = 'fc1')
positive = tf.contrib.layers.fully_connected(positive_codes, 1024, activation_fn=None, scope = 'fc2')
negative = tf.contrib.layers.fully_connected(negative_codes, 1024, activation_fn=None, scope = 'fc3')

loss, positive_dist, negative_dist = triplet_loss([anchor, positive, negative])
#optimizer = tf.train.AdamOptimizer().minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)


batch_size = 16
epochs = 10
iteration = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        for row in batch_data(X_train, batch_size):
            anc_images = row[:, 0]
            pos_images = row[:, 1]
            neg_images = row[:, 2]

            print('anc_images', anc_images.shape, anc_images.dtype)
            print('pos_images', pos_images.shape, pos_images.dtype)
            print('neg_images', neg_images.shape, neg_images.dtype)

            feed = {anchor: anc_images, positive: pos_images, negative: neg_images}
            loss, _ = sess.run([loss, optimizer], feed_dict=feed)
            print(
                "Epoch: {}/{}".format(e+1, epochs),
                "Iteration: {}".format(iteration),
                "Training loss: {:.5f}".format(loss),
                #"Positive dist: {:.5f}".format(p_dist),
                #"Negative dist: {:.5f}".format(n_dist)
            )
            iteration += 1