import tempfile
import uuid
from io import BytesIO, StringIO

from os import makedirs
from os.path import join

from PIL import Image
from flask import request


def get_bytes_from_base64_image(base64, image_type='image/jpeg'):
    img_data = base64.replace('data:{};base64'.format(image_type), '')
    return BytesIO(base64.b64decode(img_data))


def make_tmp_image(image_bytes):
    with tempfile.NamedTemporaryFile(mode="wb") as jpg:
        jpg.write(image_bytes.getbuffer())
    return jpg


def get_tmp_image_from_base64(base64, image_type='image/jpeg'):
    return make_tmp_image(get_bytes_from_base64_image(base64, image_type))


def save_image_from_base64(base64, path, image_type='image/jpeg'):
    filename = str(uuid.uuid4()) + '.' + image_type.replace('image/', '')
    makedirs(path, exist_ok=True)
    file_path = join(path, filename)
    image_bytes = get_bytes_from_base64_image(base64, image_type)
    image = Image.open(image_bytes)
    image.save(file_path)
    return image


def request_get(name):
    return request.form.get(name, None)


def request_has(name):
    return name in request.form


def request_dict(*args, default=None):
    if len(args):
        return { arg: request.form.get(arg, default) for arg in args }

    return request.form
