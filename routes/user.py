import numpy as np
from flask import flash, json, url_for
from flask import redirect
from flask import render_template
from flask import request
from flask import session

from acme.auth import Auth, is_logged_in
from acme.url import URL
from acme.utils import get_tmp_image_from_base64, save_image_from_base64, request_get, request_has
from app import app, db, face_net_instance
from models.user import ProfileImages, User


@app.route('/user/<int:id>',methods=['GET'])
def user(id):
    user = User.query.filter_by(id=id).first()
    return render_template('users/show.html', user=user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if Auth.check():
        return redirect('/')

    if request.method == 'POST':

        try:
            email = request.form['email']

            if request.form['face_photo']:
                jpg = get_tmp_image_from_base64(request.form['face_photo'])
                preprocessed_image = face_net_instance.process_image(jpg.name, 180)
                if preprocessed_image is not None:
                    Auth.verify_by_image(email, preprocessed_image)
                else:
                    flash('Please, put your face straight at the webcam!', 'danger')
            else:
                password = request.form['password']
                user = Auth.verify(email, password)

                if user:
                    Auth.save(user)
                    flash('Hi {} {}'.format(user.first_name, user.last_name), 'success')
                    return redirect(URL.back())

                flash('Wrong email or password', 'danger')
        except Exception as e:
            flash(str(e), 'danger')

    return render_template('auth/login.html')


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    user_form = {}

    if request.method == 'POST':
        try:
            first_name = request.form['first_name']
            last_name = request.form['last_name']
            email = request.form['email']
            password = request.form['password']
            user_form = request.form

            if password != request.form['confirm_password']:
                raise Exception('Please confirm your password')

            user = User(first_name=first_name, last_name=last_name, email=email, password=Auth.crypt_password(password))
            db.session.add(user)
            db.session.commit()
            Auth.save(user)
            return redirect('/')
        except Exception as e:
            flash(str(e), 'danger')

    return render_template('auth/registration.html', user_form=user_form)


@app.route('/logout')
def logout():
    session.clear()
    flash('You are now logget out', 'success')
    return redirect('/')


@app.route('/profile', methods=['GET', 'POST'])
@is_logged_in
def profile():
    current_user = Auth.user()

    if request.method == 'POST':
        try:
            current_user.first_name = request_get('first_name')
            current_user.last_name = request_get('last_name')
            current_user.email = request_get('email')

            if request_has('password'):
                current_user.password = Auth.crypt_password(request_get('password'))

                if request_get('password') != request_get('confirm_password'):
                    raise Exception('Please confirm your password')

            db.session.commit()
        except Exception as e:
            flash(str(e), 'danger')

    return render_template('users/edit.html', user=current_user)


@app.route('/upload_profile_image', methods=['POST'])
@is_logged_in
def upload_profile_image():
    if request.method == 'POST':
        user = Auth.user()

        image = save_image_from_base64(request.form['fileToUpload'], user.get_profile_images_dir())

        preprocessed_image = face_net_instance.process_image(image.filename, 180)

        if preprocessed_image is not None:
            profile_images = ProfileImages(src=image.filename, user_id=Auth.user().id)
            db.session.add(profile_images)
            db.session.commit()
            try:
                user_wights = np.load(user.get_profile_images_weights() + '.npy')
                np.append(user_wights, {profile_images.id: preprocessed_image})
            except FileNotFoundError:
                user_wights = {profile_images.id: preprocessed_image}
            np.save(user.get_profile_images_weights(), user_wights)

            return json.dumps({
                'success': True,
                'path': url_for('static', filename=image.filename)
            })
    return json.dumps({
        'success': False
    })