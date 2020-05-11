import logging
from os import urandom
from tempfile import mkdtemp

import eventlet
import sys

from flask_session import Session
from flask_socketio import SocketIO
from acme.Networks.FaceNet.face_net import FaceNet
import boto3

eventlet.monkey_patch(thread=False)

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os.path import abspath, dirname, join, isfile

root = logging.getLogger()
root.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
root.addHandler(ch)

APP_PATH = dirname(abspath(__file__))
APP_STATIC = join(APP_PATH, 'static')
APP_URL = '/'

app = Flask(__name__)
app.secret_key = urandom(32)
app.config.from_pyfile('config.cfg')
db = SQLAlchemy(app)
s3 = boto3.client('s3',
                  endpoint_url=app.config['S3_HOST'],
                  use_ssl=app.config['S3_USE_SSL'],
                  aws_access_key_id=app.config['S3_AWS_ACCESS_KEY_ID'],
                  aws_secret_access_key=app.config['S3_AWS_SECRET_ACCESS_KEY'],
                  region_name=app.config['S3_REGION_NAME'])

#sess = Session()
#app.config['SESSION_TYPE'] = 'filesystem'#filesystem
#app.config["SESSION_FILE_DIR"] = 'D:\Python\\flask\\boto3_pytest_docker' #mkdtemp()
#sess.init_app(app)

#mgr = socketio.KombuManager('redis://172.96.50.3', write_only=True)
#sio = socketio.Server(client_manager=mgr)
socketio_app = SocketIO(app, message_queue='redis://{}'.format(app.config['REDIS_IP']))

face_net_instance = FaceNet()
if isfile(join(APP_PATH, app.config['FACE_NET_LANDMARKS_FILE'])):
    face_net_instance.set_align_dlib_path(join(APP_PATH, app.config['FACE_NET_LANDMARKS_FILE']))

from routes.dashboard import *
from routes.receipt import *
from routes.user import *
from routes.face_net_dashboard import *
import commands

if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    #default run application
    #app.run(debug=True, host='127.0.0.1')

    #run application with flask_socketio
    socketio_app.run(app, host='127.0.0.1', port=8000)#0.0.0.0 for docker binding

    # wrap Flask application with socketio's middleware
    #app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    #eventlet.wsgi.server(eventlet.listen(('', 8000)), app)
