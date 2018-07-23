import requests
import sqlite3
from contextlib import contextmanager
import uuid
import pandas as pd
from joblib import Parallel, delayed
from lxml import html
import json
import os

DATABASE_LOCATION = 'data/html.db'
IMAGE_DIR = 'images'


def get_tmp_table_name():
    return '__tmp_' + str(uuid.uuid4())


@contextmanager
def tmp_table(df, conn):
    name = get_tmp_table_name()
    df.to_sql(name, conn, index=False)
    try:
        yield name
    finally:
        cur = conn.cursor()
        cur.execute(f'drop table if exists `{name}`')


def insert_into_table(df, table_name, conn, replace=False):
    cur = conn.cursor()
    tmp_table_name = '__tmp_' + str(uuid.uuid4())
    try:
        cur.execute(f'create temporary table `{tmp_table_name}` as select * from `{table_name}` where 0;')
        df.to_sql(tmp_table_name, conn, if_exists='append', index=False)
        cur.execute(('replace' if replace else 'insert or ignore') + f' into `{table_name}` select * from `{tmp_table_name}`')
    finally:
        cur.execute(f'drop table if exists `{tmp_table_name}`;')


def initialize_database():
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS `html` (
                `url`	TEXT,
                `timestamp`	DATETIME DEFAULT CURRENT_TIMESTAMP,
                `html`	TEXT NOT NULL,
                PRIMARY KEY(`url`,`timestamp`)
            );""")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS `post` (
                `id`	INTEGER NOT NULL,
                `created_at`	INTEGER,
                `creator_id`	INTEGER,
                `author`	TEXT,
                `change`	INTEGER,
                `source`	TEXT,
                `score`	INTEGER,
                `md5`	TEXT,
                `file_size`	INTEGER,
                `file_url`	TEXT,
                `is_shown_in_index`	BOOLEAN,
                `preview_url`	TEXT,
                `preview_width`	INTEGER,
                `preview_height`	INTEGER,
                `actual_preview_width`	INTEGER,
                `actual_preview_height`	INTEGER,
                `sample_url`	TEXT,
                `sample_width`	INTEGER,
                `sample_height`	INTEGER,
                `sample_file_size`	INTEGER,
                `jpeg_url`	TEXT,
                `jpeg_width`	INTEGER,
                `jpeg_height`	INTEGER,
                `jpeg_file_size`	INTEGER,
                `rating`	INTEGER,
                `has_children`	BOOLEAN,
                `parent_id`	INTEGER,
                `status`	TEXT,
                `width`	INTEGER,
                `height`	INTEGER,
                `is_held`	BOOLEAN,
                `frames_pending_string`	TEXT,
                `frames_string`	INTEGER
            );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS `image_tag` (
            `image_id`	INTEGER,
            `name`	TEXT,
            PRIMARY KEY(`image_id`,`name`)
        );""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS `tag` (
            `name`	TEXT,
            `type`	TEXT,
            PRIMARY KEY(`name`)
        );""")


def update_missing_or_expired_urls(urls, connections=1, update_after_days=1.0):
    urls_df = pd.DataFrame(urls, columns=['url'])
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        with tmp_table(urls_df, db) as tmp_table_name:
            cur.execute(
                f"""select t1.url from `{tmp_table_name}` as t1 join html on t1.url = html.url
                    group by html.url
                    having julianday(current_timestamp) - max(julianday(html.timestamp)) >= {update_after_days}
                    UNION
                    select t1.url from `{tmp_table_name}` as t1 left join html on t1.url = html.url
                    where html.timestamp is null
                    """)
            urls_to_get = [x[0] for x in cur.fetchall()]
            responses = Parallel(n_jobs=connections)(delayed(requests.get)(url) for url in urls_to_get)
            cur.executemany('insert into html (url, html) values (?, ?)',
                            [(url, r.content) for url, r in zip(urls, responses)])


def get_html(urls):
    """

    :param urls: list of urls to try to get the html from the database for; only gets those in the database
    :return: list of tuples (url, html); html is binary
    """
    urls_df = pd.DataFrame(urls, columns=['url'])
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        with tmp_table(urls_df, db) as tmp_table_name:
            cur.execute(f"""
            select url, html from html 
                natural join (select url, max(timestamp) as timestamp from html group by url)
                natural join `{tmp_table_name}`""")
            return cur.fetchall()


def chunks(l, chunk_size):
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def get_tag_and_post_data_from_html(html_text):
    tree = html.fromstring(html_text)
    tag_data = dict()
    post_data = dict()
    for i, script_elem in enumerate(tree.xpath('//script')):
        if script_elem is None:
            continue
        if "Post.register(" in str(script_elem.text):
            for line in script_elem.text.split('\n'):
                line = line.strip()
                if line == '':
                    continue
                if line.startswith('Post.register_tags('):
                    line = line[len('Post.register_tags('):].rstrip(');')
                    tag_data.update(json.loads(line))
                elif line.startswith('Post.register('):
                    line = line[len('Post.register('):].rstrip(');')
                    post_dict = json.loads(line)
                    assert post_dict['id'] not in post_data
                    post_data[post_dict['id']] = post_dict
    return tag_data, post_data


def download_post(file_url, post_id):
    extension = file_url[file_url.rindex('.'):]
    filename = os.path.join(IMAGE_DIR, str(post_id) + extension)
    assert not os.path.exists(filename)
    with open(filename, 'wb') as f:
        r = requests.get(file_url)
        f.write(r.content)
    print(post_id)


def download_images():
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        existing_images = os.listdir(IMAGE_DIR)
        existing_post_ids = [(x[:x.rindex('.')],) for x in existing_images]
        tmp_table_name = get_tmp_table_name()
        cur.execute(f"""
            CREATE TEMPORARY TABLE `{tmp_table_name}` (
                `post_id`	INTEGER,
                PRIMARY KEY(`post_id`)
            );""")
        cur.executemany(f'insert into `{tmp_table_name}` values (?)', existing_post_ids)
        cur.execute(f'select id, file_url From post where not exists(select 1 from `{tmp_table_name}` as t1 where t1.post_id = post.id)')
        id_file_url_list = cur.fetchall()
    Parallel(n_jobs=4)(delayed(download_post)(file_url, post_id) for post_id, file_url in id_file_url_list)


def run():
    # urls = [
    #     'http://konachan.com/post',
    #     'http://konachan.com/post?page=2',
    #     'http://konachan.com/post?page=3',
    # ]
    # update_missing_or_expired_urls(urls=urls)
    # get_html(urls[:1])
    urls = [f'http://konachan.com/post?page={x}' for x in range(9939, 0, -1)]

    # for chunk in chunks(urls, 150):
    #     print(chunk)
    #     update_missing_or_expired_urls(chunk, connections=7)
    url_html_list = get_html(urls)
    tag_post_dict_list = Parallel(n_jobs=-2)(delayed(get_tag_and_post_data_from_html)(html_text) for _, html_text in url_html_list)

    complete_post_dict = dict()
    complete_tag_dict = dict()
    for tag_dict, post_dict in tag_post_dict_list:
        complete_post_dict.update(post_dict)
        complete_tag_dict.update(tag_dict)

    df = pd.DataFrame(complete_post_dict).T
    df.sort_index(inplace=True)
    del df['frames_pending']
    del df['frames']
    post_tag_list = []
    for post_id, tags in df['tags'].iteritems():
        post_tag_list += [(post_id, tag) for tag in tags.split(' ')]
    del df['tags']
    with sqlite3.connect(DATABASE_LOCATION) as db:
        cur = db.cursor()
        cur.executemany('insert OR IGNORE into image_tag values (?, ?)', post_tag_list)
        insert_into_table(df, table_name='post', conn=db, replace=False)

    with open('tmp.json', 'w') as f:
        json.dump(tag_post_dict_list, f)


if __name__ == '__main__':
    download_images()
