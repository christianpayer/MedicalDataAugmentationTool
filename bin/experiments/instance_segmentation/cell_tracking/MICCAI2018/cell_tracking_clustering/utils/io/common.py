
import os


def create_directories_for_file_name(file_name):
    dir_name = os.path.dirname(file_name)
    create_directories(dir_name)


def create_directories(dir_name):
    if dir_name == '':
        return
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)