from acme.Networks.FRNN.fr_utils import get_batch_from_peoples, preprocess_images, preprocess_image
from acme.Networks.FRNN.fr_utils import list_dir
from acme.Networks.FRNN.inception_blocks_v2 import faceRecoModel
import tensorflow as tf
import os
import numpy as np

dir = '/home/srivoknovskiy/deepnets/lfw'
peoples = list_dir(dir)
dataset = []
imw = 96
imh = 96
for i,p,n in get_batch_from_peoples(peoples):
    anchor = os.path.join(dir, i)
    positive = os.path.join(dir, p)
    negative = os.path.join(dir, n)
    dataset.append([
        preprocess_image(anchor, size=(96,96)),
        preprocess_image(positive, size=(96,96)),
        preprocess_image(negative, size=(96,96)),
    ])

X_train = np.array(dataset)
print(X_train.shape)
# raise EOFError

def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.maximum(tf.reduce_mean(basic_loss), 0.0), pos_dist, neg_dist

    with tf.control_dependencies([y_true]):
        value = tf.identity(loss)
        return value

FRmodel = faceRecoModel(input_shape=(3, 96, 96))
print("Total Params:", FRmodel.count_params())

#http://www.lambdatwist.com/concatenating-metadata-with-keras-embeddings/

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])


FRmodel.fit(X_train, batch_size=10, epochs=10, verbose=1, validation_split=0.1)

#load_weights_from_FaceNet(FRmodel)
#FRmodel.load_weights('/home/srivoknovskiy/Python/flask/acme/Networks/FRNN/weights/vgg-face-keras.h5')
#model.save_weights('my_model_weights.h5')
