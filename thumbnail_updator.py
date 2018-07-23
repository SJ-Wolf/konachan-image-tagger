import pandas as pd
import sqlite3
from PIL import Image
from main import DATABASE_LOCATION, IMAGE_DIR, chunks
import numpy as np
import os
from joblib import Parallel, delayed
from time import time

Image.MAX_IMAGE_PIXELS *= 5


def get_square_thumb_from_filename(filename, size, method='pad'):
    try:
        img = Image.open(filename)
        img = image_to_square(img, method=method)
        img = img.resize((size, size))
    except OSError as e:
        print(e)
        return None
    # img.save('sample_thumb.jpg')
    return img


def get_rgb_matrix_from_img(img):
    if img is None:
        return None
    if img.mode == 'RGB':
        return np.asarray(img)
    elif img.mode == 'L' or img.mode == 'P':
        return np.stack((np.asarray(img),) * 3, -1)
    elif img.mode == 'LA':
        return np.stack((np.asarray(img)[:, :, :1],) * 3, -1)
    elif img.mode == 'RGBA':
        return np.asarray(img)[:, :, :3]
    elif img.mode == 'CMYK':
        img = img.convert('RGB')
        return np.asarray(img)
    else:
        raise Exception(f"Unknown image mode: {img.mode}.")


def image_to_square(img, method='crop'):
    if method == 'crop':
        longer_side = min(img.size)
    elif method == 'pad':
        longer_side = max(img.size)
    else:
        raise Exception("Method not found: " + method)

    horizontal_padding = (longer_side - img.size[0]) / 2
    vertical_padding = (longer_side - img.size[1]) / 2
    img5 = img.crop(
        (
            -horizontal_padding,
            -vertical_padding,
            img.size[0] + horizontal_padding,
            img.size[1] + vertical_padding
        )
    )
    return img5


def get_thumbnail_blob_from_filename(filename, size, method='pad'):
    img = get_square_thumb_from_filename(filename, size, method)
    if img is None:
        return None
    arr = get_rgb_matrix_from_img(img)
    return arr.tostring()


def update_missing_thumbnails(size=200, method='pad', chunk_size=2000):
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        cur.execute(f"""
            select id, file_url from post
            where not exists(select 1 from thumbnail where thumbnail.post_id = post.id and thumbnail.method = '{method}' and thumbnail.size = {size})
            order by post.id asc
            """)
        for chunk in chunks(cur.fetchall(), chunk_size):
            id_file_url_list = chunk
            post_ids = [x[0] for x in id_file_url_list]
            file_names = [os.path.join(IMAGE_DIR, str(x[0]) + x[1][x[1].rindex('.'):]) for x in id_file_url_list]
            print(post_ids[0])
            method = 'pad'
            size = 200
            thumbnail_blob_list = Parallel(n_jobs=-3)(delayed(get_thumbnail_blob_from_filename)(file_name, size, method) for file_name in file_names)
            cur.executemany('insert or ignore into thumbnail (post_id, size, method, thumbnail) values (?, ?, ?, ?)',
                            [(post_id, size, method, thumbnail_blob) for post_id, thumbnail_blob in zip(post_ids, thumbnail_blob_list) if
                             thumbnail_blob is not None])
            db.commit()


if __name__ == '__main__':
    update_missing_thumbnails(200, 'pad', chunk_size=2000)
