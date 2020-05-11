import sqlite3
import threading
from itertools import zip_longest
from multiprocessing import Pool
from multiprocessing.dummy import Pool as dummy_Pool
from os.path import abspath, dirname, join, isfile, basename

import time
import os
import pandas as pd

from comments_repository import CommentsRepository
from sqlite_storage import SQLiteStorage
from tqdm import trange, tqdm

start = time.time()

tic = lambda start_time=start: 'at %8.4f seconds' % (time.time() - start_time)
db_folder = 'D:\\7     Network\ChatBot\db'
files_folder = 'D:\\7     Network\ChatBot\set'
#db_folder = 'D:\datasets\db'
#files_folder = 'C:\set'

log_file = 'process_thread_log.txt'
global_to_file = 'C:\\train.to'
global_from_file = 'C:\\train.from'

def get_databases(dir):
    files = os.listdir(dir)
    for file in files:
        if not file.endswith('.db'): continue
        if not isfile(join(files_folder, get_db_name(file)+'.from')):
            yield file
        else:
            print(file, 'are exists in', db_folder)

def make_training_set():
    total = len(os.listdir(db_folder))# hack. Can cause some problems
    number = 0

    start_total_time = time.time()

    #with tqdm(total=total) as pbar:
    with Pool(processes=os.cpu_count()) as pool:
        start_time = time.time()
        files = pool.map(parse_db_to_file, get_databases(db_folder), 1)
        number += len(files)
        #pbar.update(len(files))
        log("Proccessed", number, 'of', total, 'by', tic(start_time))

    print('Total time execution', tic(start_total_time))

def parse_db_to_file(db):
    if not db: return
    start_time = time.time()
    name = get_db_name(db)
    repository = CommentsRepository(SQLiteStorage(name, db_folder))

    limit = 10000
    last_unix = 0
    cur_length = limit
    counter = 0
    rows = 0

    from_file = join(files_folder, name+'.from')
    to_file = join(files_folder, name+'.to')

    while cur_length == limit:
        df = repository.get_batch(last_unix, limit)
        cur_length = len(df)
        rows += cur_length

        try:
            last_unix = df.tail(1)['unix'].values[0]
        except:
            continue

        write_to_file(from_file, df['parent'].values)
        write_to_file(to_file, df['comment'].values)

        counter += 1
        if counter % 100 == 0:
            log(get_name(), 'Proccessing rows', rows, tic(start_time))

    log(get_name(), 'Finish parse', 'rows', rows, tic(start_time), name)


def iterate_by_batch(array_list, amount, fillvalue=None):
    args = [iter(array_list)] * amount
    return zip_longest(*args, fillvalue=fillvalue)


def write_to_file(file, data):
    with open(file, 'a', encoding='utf8') as f:
        for content in data:
            f.write(content+'\n')


def log(*args):
    print(*args)
    try:
        with open(log_file, 'a', encoding='utf8') as f:
            for content in args:
                f.write(str(content)+' ')
            f.write('\n')
    except Exception as e:
        print('ERROR save logs to file', str(e))


def get_name():
    return "Proc {}: ".format(os.getpid())
    #return "Proc {}, {}: ".format(os.getpid(), threading.current_thread().name)


def get_db_name(file):
    return file.replace('.db', '')

def concatenate_files():
    thread1 = threading.Thread(target = parse_from_files, args = ())
    thread2 = threading.Thread(target = parse_to_files, args = ())
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    print('Finished')

def parse_from_files():
    with open(global_from_file, 'a', encoding='utf8') as f_from:
        for file in os.listdir(files_folder):
            if not file.endswith('.from'): continue
            file_path = os.path.join(files_folder, file)
            print('processing', file)
            with open(file_path, 'r', encoding='utf8') as f:
                f_from.writelines(f.readlines())

def parse_to_files():
    with open(global_to_file, 'a', encoding='utf8') as f_to:
        for file in os.listdir(files_folder):
            if not file.endswith('.to'): continue
            file_path = os.path.join(files_folder, file)
            print('processing', file)
            with open(file_path, 'r', encoding='utf8') as f:
                f_to.writelines(f.readlines())

def cehck():
    c=0
    with open(global_to_file, 'r', encoding='utf8') as f_to:
        for line in f_to.readlines():
            print(line, end='')
            c+=1
            if c > 1000: break


if __name__ == '__main__':
    #make_training_set()
    concatenate_files()


