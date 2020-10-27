
import os
import shutil
import sys


def create_directories_for_file_name(file_name):
    """
    Create missing directories for the given file name.
    :param file_name: The file name.
    """
    dir_name = os.path.dirname(file_name)
    create_directories(dir_name)


def create_directories(dir_name):
    """
    Create missing directories.
    :param dir_name: The directory name.
    """
    if dir_name == '':
        return
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def copy_files_to_folder(files, dir_name):
    """
    Copy files to a directory.
    :param files: List of files to copy.
    :param dir_name: The directory name.
    """
    create_directories(dir_name)
    for file_to_copy in files:
        shutil.copy(file_to_copy, dir_name)


class Tee(object):
    """
    Object that can write to multiple files at a time.
    """
    def __init__(self, *files):
        """
        Initializer.
        :param files: List of files to write to.
        """
        self.files = files

    def write(self, obj):
        """
        Write object to files.
        :param obj: The object to write.
        """
        for f in self.files:
            f.write(obj)

    def flush(self):
        """
        Flush file objects.
        """
        for f in self.files:
            f.flush()


class RedirectStreamToFile(object):
    """
    Class that redirects stdout or stderr additionally to a file. Most of the code is taken from contextlib.redirect_stdout.
    """
    _stream = None

    def __init__(self, filename):
        self._filename = filename
        self._redirect_file = None
        self._new_target = None
        # We use a list of old targets to make this CM re-entrant
        self._old_targets = []

    def __enter__(self):
        self._redirect_file = open(self._filename, 'w')
        self._new_target = Tee(getattr(sys, self._stream), self._redirect_file)
        self._old_targets.append(getattr(sys, self._stream))
        setattr(sys, self._stream, self._new_target)
        return self._new_target

    def __exit__(self, exctype, excinst, exctb):
        setattr(sys, self._stream, self._old_targets.pop())


class redirect_stdout_to_file(RedirectStreamToFile):
    """
    Redirects stdout to both stdout and a file.
    """
    _stream = 'stdout'


class redirect_stderr_to_file(RedirectStreamToFile):
    """
    Redirects stderr to both stderr and a file.
    """
    _stream = 'stderr'
