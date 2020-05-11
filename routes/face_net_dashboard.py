import base64
import bz2
import os
import tarfile
import urllib.request

import numpy as np
from flask import render_template

from acme.Networks.FaceNet.download_and_extract_model import download_and_extract_model
from acme.Networks.FaceNet.preprocess import preprocess
from acme.Networks.FaceNet.test_classifier import get_emmbedings
from acme.auth import Auth, is_logged_in
from acme.url import absolute
from app import app, face_net_instance, socketio_app


@app.route('/face_net_dashboard', methods=['GET', 'POST'])
@is_logged_in
def face_net_dashboard():
    return render_template('admin/face_net_dashboard.html', user=Auth.user())


@socketio_app.on('check')
def check_request(message):
    name = message['name']

    try:
        info = None

        if name == 'landmarks': info = get_landmark_info()
        if name == 'model': info = get_model_info()
        if name == 'lfw': info = get_lfw_info()
        if name == 'output': info = get_output_info()
        #if name == 'prediction': return

        if info:
            socketio_app.emit('finish-' + str(name), info)

    except Exception as e:
        socketio_app.emit('finish-' + str(name), {'error': "{}: {}".format(type(e).__name__, str(e))})


@socketio_app.on('update')
def update_request(message):
    print('Receive', message)
    name = message['name']

    try:
        if name == 'landmarks': update_landmark()
        if name == 'model': update_model()
        if name == 'lfw': update_lfw()
        if name == 'output': update_output()
        if name == 'prediction': make_tests()
    except Exception as e:
        socketio_app.emit('finish-' + str(name), {'error': "{}: {}".format(type(e).__name__, str(e))})


def update_model():
    socketio_app.emit('log-model', {'message': 'Start updating...'})
    download_and_extract_model('20170511-185253', absolute(app.config['FACE_NET_WEIGHTS_DIR']))
    socketio_app.emit('log-model', {'message': 'Done'})
    socketio_app.emit('finish-model', get_model_info())


def get_model_info():
    dir = absolute(app.config['FACE_NET_WEIGHTS_DIR'])

    if not os.path.isdir(dir): return {'error': 'Folder does not exists'}

    return {
        'path': dir,
        'size': sizeof_fmt(get_size(dir))
    }


def update_landmark():
    socketio_app.emit('log-landmark', {'message': 'Start downloading...'})
    file = absolute(app.config['FACE_NET_LANDMARKS_FILE'])
    archive = file + '.bz2'
    urllib.request.urlretrieve(app.config['FACE_NET_LANDMARKS_URL'], archive)
    socketio_app.emit('log-landmark', {'message': 'Extracting...'})

    with open(file, 'wb') as new_file, bz2.BZ2File(archive, 'rb') as bz2_file:
        for bytes in iter(lambda : bz2_file.read(100 * 1024), b''):
        #for bytes in file.read():
            new_file.write(bytes)

    os.remove(archive)
    socketio_app.emit('log-landmark', {'message': 'Done'})
    socketio_app.emit('finish-landmark', get_landmark_info())


def get_landmark_info():
    file = absolute(app.config['FACE_NET_LANDMARKS_FILE'])

    if not os.path.isfile(file): return {'error': 'File does not exists'}

    return {
        'path': file,
        'size': sizeof_fmt(os.path.getsize(file))
    }


def update_lfw():
    socketio_app.emit('log-lfw', {'message': 'Start downloading...'})
    archive = absolute(app.config['FACE_NET_DATA_DIR']) + '.tar.gz'
    data = absolute(app.config['FACE_NET_DATA_DIR'])
    urllib.request.urlretrieve(app.config['FACE_NET_LWF_URL'], archive)
    socketio_app.emit('log-lfw', {'message': 'Extracting...'})
    os.makedirs(data, exist_ok=True)
    tar = tarfile.open(archive, "r:gz")
    tar.extractall(data)
    tar.close()
    os.remove(archive)
    socketio_app.emit('log-lfw', {'message': 'Done'})
    socketio_app.emit('finish-lfw', get_lfw_info())


def get_lfw_info():
    data = absolute(app.config['FACE_NET_DATA_DIR'])
    return {
        'path': data,
        'size': sizeof_fmt(get_size(data))
    }


def update_output():
    socketio_app.emit('log-output', {'message': 'Start updating...'})
    output_dir = absolute(app.config['FACE_NET_OUTPUT_DIR'])
    preprocess(absolute(app.config['FACE_NET_DATA_DIR']), output_dir, 180)
    socketio_app.emit('log-output', {'message': 'Done'})
    socketio_app.emit('finish-output', get_output_info())


def get_output_info():
    dir = absolute(app.config['FACE_NET_OUTPUT_DIR'])
    size = get_size(dir)

    if size == 0:
        return {'error': 'Images are not preprocessed'}

    return {
        'path': dir,
        'size': sizeof_fmt(size)
    }


def make_tests():
    crop_dim = 180
    print('make_tests')
    socketio_app.emit('log-prediction', {'message': 'Start making tests...'})
    im1 = '/home/srivoknovski/Python/flask/acme/Networks/FaceNet/data/lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg'
    im2 = '/home/srivoknovski/Python/flask/acme/Networks/FaceNet/data/lfw/Aaron_Peirsol/Aaron_Peirsol_0004.jpg'
    im3 = '/home/srivoknovski/Python/flask/acme/Networks/FaceNet/data/lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg'

    with open(im1, "rb") as image_file: socketio_app.emit('log-prediction', {'image': (image_file.read())})
    with open(im2, "rb") as image_file: socketio_app.emit('log-prediction', {'image': (image_file.read())})
    with open(im3, "rb") as image_file: socketio_app.emit('log-prediction', {'image': (image_file.read())})

    socketio_app.emit('log-prediction', {'message': 'preprocessing test images..., Crop dimension {}'.format(crop_dim)})
    images = []
    images.append(face_net_instance.process_image(im1, crop_dim))
    images.append(face_net_instance.process_image(im2, crop_dim))
    images.append(face_net_instance.process_image(im3, crop_dim))

    socketio_app.emit('log-prediction', {'message': 'loading model...'})
    model_path = absolute(app.config['FACE_NET_WEIGHTS_FILE'])
    embs = get_emmbedings(images=images, model_path=model_path)
    socketio_app.emit('log-prediction', {'message': 'Model path {} {}'.format(model_path, os.path.getsize(model_path))})

    diff1 = np.linalg.norm(embs[0] - embs[1])
    diff2 = np.linalg.norm(embs[0] - embs[2])

    print(im1, im2, np.linalg.norm(embs[0] - embs[1]))
    print(im1, im3, np.linalg.norm(embs[0] - embs[2]))
    socketio_app.emit('log-prediction', {'message': 'Done'})
    socketio_app.emit('finish-prediction', {
        'The same persons': str(diff1),
        'The different persons': str(diff2)
    })


def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)