import bz2
import sqlite3
import json
import tarfile
import threading
from datetime import datetime

import os

import shutil

#from acme.Networks.ChatBot.comments_repository import CommentsRepository
#from acme.Networks.ChatBot.sqlite_storage import SQLiteStorage
#from acme.Networks.ChatBot.win32_utils import FreeSpace
from itertools import zip_longest
from multiprocessing.pool import Pool
from time import sleep

from comments_repository import CommentsRepository
from sqlite_storage import SQLiteStorage
from win32_utils import FreeSpace

db_folder = 'C:\db'
archives = []
is_end = False
num_runned_threads = 0

def parse_comments():
    folders_with_comments = [
        'E:\dataset\\2014',
        'E:\dataset\\2015',
        'E:\dataset\\2016',
        'D:\datasets\\2007',
        'D:\datasets\\2008',
        'D:\datasets\\2009',
        'D:\datasets\\2010',
        'D:\datasets\\2011',
        'D:\datasets\\2012',
        'D:\datasets\\2013',
    ]
    for folder in folders_with_comments:
        read_folder(folder)

    start_parse()

def read_folder(dir):
    files = os.listdir(dir)
    for file in files:
        if not file.endswith('bz2'):
            continue
        archive = os.path.join(dir, file)
        #read_bzfile(archive, file.replace('.bz2', ''))
        name = file.replace('.bz2', '')
        if not db_exists(name):
            archives.append((archive, name))

def read_bzfile(item):
    if not item: return

    bzfile, name = item
    #if not db_exists(name):
    print(get_process_name(), 'Read Archive', bzfile, 'name', name)
    source_file = bz2.BZ2File(bzfile, "r")
    parse_comment(source_file, name)

    return name

def start_parse():
    global archives, num_runned_threads
    config = get_config('config.json')
    total = len(archives)
    processed = 0
    all_files = []

    with Pool(processes=config['max_threads']) as pool:
        for batch_files in iterate_by_batch(archives, config['max_threads'], None):
            files = pool.map(read_bzfile, batch_files, 1)
            processed += len(files)
            print('Processed', files, ',', processed, 'of', total)
            all_files += files

def db_exists(name):
    if os.path.isfile('C:/db/{}.db'.format(name)):
        print(get_process_name(), 'File ', name, 'Exists in C:/db/')
        return True

    if os.path.isfile('D:/datasets/db/{}.db'.format(name)):
        print(get_process_name(), 'File ', name, 'Exists in D:/datasets/db')
        return True

    print(get_process_name(), 'File', name, 'Doesnt Exists')
    return False


def get_config(filename):
    with open(filename) as f:
        return json.loads(f.read())


def iterate_by_batch(array_list, amount, fillvalue=None):
    args = [iter(array_list)] * amount
    return zip_longest(*args, fillvalue=fillvalue)


def parse_comment(file, db_name):
    total_rows = 0
    parsed_rows = 0
    replaced_rows = 0
    print(get_process_name(), 'Create db', db_name, 'in', db_folder)
    repository = CommentsRepository(SQLiteStorage(db_name, db_folder=db_folder))
    repository.create_table()

    for row in file:
        total_rows += 1
        try:
            row = json.loads(row)
            comment_id = row.get('name', 't1_' + row.get('id'))
            parent_id = row['parent_id']
            body = format_data(row['body'])
            created_at = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']
            parent_data = repository.find_parent_comment(parent_id)
        except Exception as e:
            print(get_process_name(), 'Exception', str(e))
            print(get_process_name(), row)
            continue

        if score >= 2 and acceptable(body):
            existing_comment_score = repository.find_existing_score(parent_id)

            if existing_comment_score:
                if score > existing_comment_score:
                    repository.replace_comment(comment_id, parent_id, parent_data, body, subreddit, created_at, score)
                    replaced_rows += 1
            else:
                if parent_data:
                    repository.insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_at, score)
                    parsed_rows += 1
                else:
                    repository.insert_no_parent(comment_id, parent_id, body, subreddit, created_at, score)

        if total_rows % 500000 == 0:
            print(get_process_name(), 'File {}, Total rows read: {}, Paired rows: {}, Time: {}'.format(db_name, total_rows, parsed_rows, str(datetime.now())))

    print(get_process_name(), 'Finish ', db_name, "Total rows read: {}, Paired rows: {}".format(total_rows, parsed_rows))

def format_data(body):
    return body.replace("\n", "<EOF>").replace("\r", "<EOF>").replace('"', "'")

def acceptable(data, treshold=50):
    if len(data.split(' ')) > treshold or len(data) < 1:
        return False

    if len(data) > 1000:
        return False

    if data == '[deleted]' or data == '[removed]':
        return False

    return True

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def move_file(file, dest):
    shutil.move(file, os.path.join(dest, os.path.basename(file)))

def check_disk_space():
    space = FreeSpace(r'C:')
    print('Check disk space', space)
    if space < 10:
        files = os.listdir(db_folder)
        #move_file(os.path.join(db_folder, files[0]), 'D:/datasets/db')
        #print('Move file', files[0], 'to D:/datasets/db')

def get_process_name():
    return "Process {}".format(os.getpid())

if __name__ == "__main__":
    parse_comments()
    #read_bzfile('/home/srivoknovski/dataset/reddit/RC_2015-01.bz2', 'RC_2015-01')