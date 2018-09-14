
import os
import shutil


def create_directories_for_file_name(file_name):
    """
    Creates missing directories for the given file name.
    :param file_name: The file name.
    """
    dir_name = os.path.dirname(file_name)
    create_directories(dir_name)


def create_directories(dir_name):
    """
    Creates missing directories.
    :param dir_name: The directory name.
    """
    if dir_name == '':
        return
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def copy_files_to_folder(files, dir_name):
    """
    Copies files to a directory.
    :param dir_name: The directory name.
    """
    create_directories(dir_name)
    for file_to_copy in files:
        shutil.copy(file_to_copy, dir_name)
