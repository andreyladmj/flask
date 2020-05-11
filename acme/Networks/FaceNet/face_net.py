import glob
import logging
import multiprocessing as mp
import os
import time
import cv2
import sys
from tensorflow.python.platform import gfile
import tensorflow as tf

from acme.Networks.FaceNet.align_dlib import AlignDlib

logger = logging.getLogger(__name__)


class FaceNet:
    align_dlib = None

    def __init__(self, path=None):
        if path:
            self.set_align_dlib_path(path)

    def set_align_dlib_path(self, path):
        self.align_dlib = AlignDlib(path)

    def preprocess(self, input_dir, output_dir, crop_dim):
        start_time = time.time()
        pool = mp.Pool(processes=mp.cpu_count())

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_dir in os.listdir(input_dir):
            image_output_dir = os.path.join(output_dir, os.path.basename(os.path.basename(image_dir)))
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

        image_paths = glob.glob(os.path.join(input_dir, '**/*.jpg'))
        for index, image_path in enumerate(image_paths):
            image_output_dir = os.path.join(output_dir, os.path.basename(os.path.dirname(image_path)))
            output_path = os.path.join(image_output_dir, os.path.basename(image_path))
            pool.apply_async(self.preprocess_image, (image_path, output_path, crop_dim))

        pool.close()
        pool.join()
        logger.info('Completed in {} seconds'.format(time.time() - start_time))

    def preprocess_image(self, input_path, output_path, crop_dim):
        """
        Detect face, align and crop :param input_path. Write output to :param output_path
        :param input_path: Path to input image
        :param output_path: Path to write processed image
        :param crop_dim: dimensions to crop image to
        """
        image = self.process_image(input_path, crop_dim)
        if image is not None:
            logger.debug('Writing processed file: {}'.format(output_path))
            cv2.imwrite(output_path, image)
        else:
            logger.warning("Skipping filename: {}".format(input_path))

    def process_image(self, filename, crop_dim):
        image = self._buffer_image(filename)

        if image is not None:
            aligned_image = self._align_image(image, crop_dim)
        else:
            raise IOError('Error buffering image: {}'.format(filename))

        return aligned_image

    def _buffer_image(self, filename):
        logger.debug('Reading image: {}'.format(filename))
        image = cv2.imread(filename, )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _align_image(self, image, crop_dim):
        bb = self.align_dlib.getLargestFaceBoundingBox(image)
        aligned = self.align_dlib.align(crop_dim, image, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        if aligned is not None:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        return aligned

    def get_emmbedings(self, images, model_path):

        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            self._load_model(model_filepath=model_path)

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embedding_layer = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            return sess.run(embedding_layer, feed_dict={images_placeholder: images, phase_train_placeholder: False})

    def _load_model(self, model_filepath):
        """
        Load frozen protobuf graph
        :param model_filepath: Path to protobuf graph
        :type model_filepath: str
        """
        model_exp = os.path.expanduser(model_filepath)
        if os.path.isfile(model_exp):
            logging.info('Model filename: %s' % model_exp)
            with gfile.FastGFile(model_exp, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        else:
            logger.error('Missing model file. Exiting')
            sys.exit(-1)
