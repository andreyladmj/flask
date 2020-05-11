from functools import wraps

from acme.Networks.FaceNet.test_classifier import get_emmbedings
from acme.url import absolute
from app import app, face_net_instance
from flask import Response
from flask import flash
from flask import redirect
from flask import session
from flask import url_for, g
import numpy as np
from passlib.handlers.sha2_crypt import sha256_crypt

from models.user import User


class Auth:
    current_user = None

    @staticmethod
    def check():
        return 'user_id' in session

    @staticmethod
    def verify(email, password):
        user = User.query.filter_by(email=email).first()

        if user and sha256_crypt.verify(password, user.password):
            return user

        return False

    @staticmethod
    def verify_by_image(email, image):
        crop_dim = 180
        user = User.query.filter_by(email=email).first()
        if not user: return False
        images = []
        images.append(image)

        try:
            user_wights = np.load(user.get_profile_images_weights() + '.npy')
        except FileNotFoundError:
            return False

        user_wights = user_wights.reshape(1)

        for i in user_wights[0]:
            images.append(face_net_instance.process_image(user_wights[0][i], crop_dim))

        model_path = absolute(app.config['FACE_NET_WEIGHTS_FILE'])
        embs = get_emmbedings(images=images, model_path=model_path)
        image_emb = embs[0]

        for emb in embs[1:]:
            print('Compare Embedings', np.linalg.norm(image_emb - emb))

        return False

    @staticmethod
    def save(user):
        session['logged_in'] = True
        session['user_id'] = user.id

    @staticmethod
    def crypt_password(password):
        return sha256_crypt.encrypt(password)

    @staticmethod
    def user():
        if 'user_id' in session:
            return User.query.filter_by(id=session['user_id']).first()

        return False

app.add_template_global(Auth, name='Auth')

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwards):
        if Auth.check():
            return f(*args, **kwards)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap