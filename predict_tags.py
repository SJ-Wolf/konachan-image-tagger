import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import sqlite3
from main import DATABASE_LOCATION
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix
import pickle
import time
# from sklearn.utils import shuffle



def get_input_and_output(size, method, min_number_of_tags_per_image=25, min_number_of_images_per_tag=1000):
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        cur.execute("""DROP TABLE IF EXISTS tmp;""")
        cur.execute("""CREATE TEMPORARY TABLE tmp (
              image_id TEXT,
              name     TEXT,
              PRIMARY KEY (image_id, name)
            );""")
        cur.execute(f"""INSERT INTO tmp SELECT t1.*
                    FROM image_tag AS t1
                      JOIN (SELECT image_id
                            FROM image_tag
                            GROUP BY image_id
                            HAVING count(*) >= {min_number_of_tags_per_image}) AS t2
                      -- images should have this many tags
                      JOIN (SELECT name
                            FROM image_tag
                            GROUP BY name
                            HAVING count(*) >= {min_number_of_images_per_tag}) AS t3
                      -- tags should have this many associated images
                      JOIN thumbnail
                      -- make sure we have a thumbnail
                      JOIN post
                        ON t1.image_id = t2.image_id AND t1.name = t3.name AND t1.image_id = thumbnail.post_id AND
                           t1.image_id = post.id
                    WHERE t1.image_id NOT IN (SELECT DISTINCT image_id
                                              FROM image_tag
                                              WHERE name IN ('tagme', 'jpeg_artifacts'))
                          AND score > 100
                          AND thumbnail.thumbnail IS NOT NULL
                          AND thumbnail.size = {size}
                          AND thumbnail.method = '{method}';""")
        cur.execute("""SELECT
              image_id,
              name
            FROM tmp;""")
        image_id_name_list = cur.fetchall()
        df = pd.DataFrame(image_id_name_list, columns=['post_id', 'tag_name'])
        df['post_id'] = df['post_id'].astype('int32', copy=False)
        post_ids = shuffle(df['post_id'].unique())
        tag_names = df['tag_name'].unique()
        with open('tag_names.txt', 'w') as f:
            f.writelines([str(x) + '\n' for x in tag_names])

        tag_to_index = dict()
        filename_to_index = dict()

        for i, tag in enumerate(tag_names):
            tag_to_index[tag] = i

        for i, filename in enumerate(post_ids):
            filename_to_index[filename] = i

        S = dok_matrix((len(post_ids), len(tag_names)), dtype='uint8')
        for index, (filename, tagname) in df.iterrows():
            S[filename_to_index[filename], tag_to_index[tagname],] = 1
        output = S.toarray()

        post_id_to_blob = pd.read_sql("""
            SELECT thumbnail.post_id, thumbnail.thumbnail
            FROM thumbnail
              JOIN tmp ON thumbnail.post_id = tmp.image_id
            GROUP BY tmp.image_id;""", db)
        post_id_to_blob.set_index(['post_id'], inplace=True)
        post_id_to_blob = post_id_to_blob['thumbnail']
        return np.stack([np.fromstring(x, dtype='uint8').reshape((size, size, 3)) for x in post_id_to_blob.loc[post_ids]]), output


def get_cached_data(refresh=False, size=200, method='pad', min_number_of_images_per_tag=9000, min_number_of_tags_per_image=10):
    if refresh:
        with open('nn_data.pickle', 'wb') as f:
            pickle.dump(get_input_and_output(size=size, method=method, min_number_of_tags_per_image=min_number_of_tags_per_image,
                                             min_number_of_images_per_tag=min_number_of_images_per_tag), f)
    with open('nn_data.pickle', 'rb') as f:
        return pickle.load(f)


def get_model(input_size, output_size):
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', activation='relu', input_shape=(input_size, input_size, 3)))
    model.add(Conv2D(32, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_size, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # return the constructed network architecture
    return model

    # nn.add(Dense(10, activation="relu", input_shape=(input_size, input_size, 3)))
    # nn.add(Flatten())
    # nn.add(Dense(output_size, activation="sigmoid"))
    # nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # return nn


if __name__ == '__main__':
    size = 200
    method = 'pad'
    nn_batch_size=128

    x_complete, y_complete = get_cached_data(refresh=False, size=200, method='pad', min_number_of_tags_per_image=20, min_number_of_images_per_tag=9000)
    y_complete = to_categorical(y_complete[:, 13], 2)
    test_split_index = int(len(x_complete) * 0.8)
    x_train = x_complete[:test_split_index]
    y_train = y_complete[:test_split_index]
    x_test = x_complete[test_split_index:]
    y_test = y_complete[test_split_index:]
    print(y_test.mean(axis=0))
    print(x_complete.shape, y_complete.shape)
    model = get_model(x_complete.shape[1], 2)
    model.summary()
    # model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))
    # gen = ImageDataGenerator(horizontal_flip=True,
    #                          vertical_flip=True,
    #                          width_shift_range=0.1,
    #                          height_shift_range=0.1,
    #                          zoom_range=0.1,
    #                          rotation_range=45,
    #                          featurewise_center=True,
    #                          samplewise_center=True,
    #                          )
    gen = ImageDataGenerator()
    # gen.fit(x_train)
    generator = gen.flow(x_train, y_train, batch_size=nn_batch_size)
    model.fit_generator(generator=generator, steps_per_epoch=len(x_train) / nn_batch_size, epochs=5, shuffle=True,
                        verbose=1, validation_data=(x_test, y_test), workers=12)
