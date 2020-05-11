import random
import os
import tensorflow as tf

from acme.Networks.FRNN import cnn
from acme.Networks.FRNN.fr_utils import load_weights_from_FaceNet, img_to_encoding, list_dir, get_batch_from_peoples, \
    preprocess_image, shuffle, batch_data, show_image, show_image_path, preprocess_images
from acme.Networks.FRNN.inception_blocks_v2 import faceRecoModel, faceRecoModel2
from acme.Networks.FRNN.triplet_loss import triplet_loss, compute_triplet_loss
from acme.Networks.FRNN.verify import verify
import numpy as np
import h5py

from acme.Networks.FRNN.vgg import vgg_face
dir = '/home/srivoknovskiy/deepnets/lfw'
# dir = 'E:\dataset\lfw/'
#dir = dir + ''
peoples = list_dir(dir)

dataset = []

imw = 96
imh = 96

for i,p,n in get_batch_from_peoples(peoples):
    anchor = os.path.join(dir, i)
    positive = os.path.join(dir, p)
    negative = os.path.join(dir, n)
    dataset.append([anchor, positive, negative])

# dataset = np.array(dataset)
# print(dataset, dataset.shape)
# raise EOFError

dataset = shuffle(np.array(dataset))


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'

n_classes = 1
epochs = 100
batch_size = 10
keep_probability = 0.5
learning_rate = 0.001

tf.reset_default_graph()


anchor_image = cnn.neural_net_image_input((imw, imh, 3), name='anchor_image')
positive_image = cnn.neural_net_image_input((imw, imh, 3), name='positive_image')
negative_image = cnn.neural_net_image_input((imw, imh, 3), name='negative_image')
keep_prob = cnn.neural_net_keep_prob_input()

cnn.init()

anchor = cnn.make_logits2(anchor_image, keep_prob, reuse=False)
positive = cnn.make_logits2(positive_image, keep_prob)
negative = cnn.make_logits2(negative_image, keep_prob)

#loss = triplet_loss([anchor, positive, negative], alpha=5)
loss, positives, negatives = triplet_loss([anchor, positive, negative])
#loss, positives, negatives = compute_triplet_loss(anchor, positive, negative, margin=0.01)
optimizer = tf.train.AdamOptimizer().minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)



print('Total count of dataset', len(dataset))
print('Training...')

def check_predictions():
    im1 = preprocess_image(dir + '/Angelina_Jolie/Angelina_Jolie_0004.jpg')
    im2 = preprocess_image(dir + '/Angelina_Jolie/Angelina_Jolie_0007.jpg')
    print('On Trained photo: ', check(im1, im2))

    im1 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg')
    im2 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg')
    print('Same photo: ', check(im1, im2))

    im1 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg')
    im2 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0002.jpg')
    print('Same person 1-2: ', check(im1, im2))

    im1 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg')
    im2 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0003.jpg')
    print('Same person 1-3: ', check(im1, im2))

    im1 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0001.jpg')
    im2 = preprocess_image(dir + '/Adolfo_Aguilar_Zinser/Adolfo_Aguilar_Zinser_0001.jpg')
    print('Different persons 1-1: ', check(im1, im2))

    im1 = preprocess_image(dir + '/Abid_Hamid_Mahmud_Al-Tikriti/Abid_Hamid_Mahmud_Al-Tikriti_0002.jpg')
    im2 = preprocess_image(dir + '/Adolfo_Aguilar_Zinser/Adolfo_Aguilar_Zinser_0002.jpg')
    print('Different persons 2-2', check(im1, im2))


def check(im1, im2):
    encodec1 = sess.run(anchor, feed_dict={anchor_image: np.array([im1]), keep_prob:1.0})
    encodec2 = sess.run(anchor, feed_dict={anchor_image: np.array([im2]), keep_prob:1.0})
    dist = np.linalg.norm(encodec1 - encodec2)
    return dist


with tf.Session() as sess:
    # Initializing the variables

    # tf.summary.scalar('loss', loss)
    # tf.summary.scalar('positives', positives)
    # tf.summary.scalar('negatives', negatives)
    # tf.summary.scalar('lr', learning_rate)
    # merged = tf.summary.merge_all()
    # tf.initialize_all_variables().run()
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batches_count = 0
        cost_sum = 0

        for row in batch_data(dataset, batch_size):
            anc_images = row[:, 0]
            pos_images = row[:, 1]
            neg_images = row[:, 2]

            # print('anc_images', anc_images[0])
            # print('pos_images', pos_images[0])
            # print('neg_images', neg_images[0])

            anc_images = preprocess_images(anc_images, size=(imw, imh))
            pos_images = preprocess_images(pos_images, size=(imw, imh))
            neg_images = preprocess_images(neg_images, size=(imw, imh))

            _, cost, p, n = sess.run([optimizer, loss, positives, negatives], feed_dict={
                anchor_image: anc_images,
                positive_image: pos_images,
                negative_image: neg_images,
                keep_prob: keep_probability})

            batches_count +=1
            cost_sum += cost
            print('Batch={0} of {1}, last cost: {2}, positive: {3}, negative: {4}'.format(batches_count, (len(dataset) // batch_size), cost, p, n))
            #check_predictions()


        #print("")
        #print('Epoch {:>2}, Batch:  '.format(epoch + 1), end='')
        #print('Cost: ', (cost_sum / batches_count), end=' ')

        # if cost_sum == 0:
        #     print('Cost is minimized. Break')
        #     break