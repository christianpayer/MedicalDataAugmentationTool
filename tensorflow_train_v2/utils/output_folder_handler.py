import os
import sys
from datetime import datetime
from glob import glob

from utils.io.common import create_directories, copy_files_to_folder, Tee


class OutputFolderHandler(object):
    """
    Class that manages output folders.
    """

    def __init__(self, base_folder, model_name=None, loss_name=None, additional_info=None, cv=None, use_timestamp=True, files_to_copy=None, stdout_file_name='output.log'):
        """
        Initializer.
        :param base_folder: The base folder.
        :param model_name: The model name.
        :param loss_name: The loss name.
        :param additional_info: Additional info appended to output_folder.
        :param cv: The cross validation.
        :param use_timestamp: If True, append a timestamp_folder.
        :param files_to_copy: The files that should be copied.
        :param stdout_file_name: If not None, redicrect stdout also to this file.
        """
        self.base_folder = base_folder
        self.cv = cv
        self.model_name = model_name
        self.loss_name = loss_name
        self.additional_info = additional_info
        self.use_timestamp = use_timestamp
        self.files_to_copy = files_to_copy
        if self.files_to_copy is None:
            self.files_to_copy = ['*.py']
        self.stdout_file_name = stdout_file_name
        self.stdout_file = None
        self.stdout_backup = None
        self.output_folder = None
        self.current_output_folder = None
        self.create_output_folder()
        self.copy_files()
        self.redirect_stdout()

    def __del__(self):
        """
        Close the handler, reset stdout to default.
        """
        self.undirect_stdout()

    def close(self):
        """
        Close the handler, reset stdout to default.
        """
        self.undirect_stdout()

    def create_output_folder(self):
        """
        Generate and create the output_folder.
        Appends a path of [self.base_folder, self.model_name, self.loss_name, self.additional_info, self.cv, self.folder_timestamp()]
        """
        path_args = [self.base_folder, self.model_name, self.loss_name, self.additional_info, self.cv]
        if self.use_timestamp:
            path_args.append(self.folder_timestamp())
        path_args = [p for p in path_args if p is not None and p != '']
        self.output_folder = os.path.join(*path_args)
        create_directories(self.output_folder)

    def redirect_stdout(self):
        """
        Redirect stdout to both a file and stdout.
        """
        if self.stdout_file_name is None:
            return
        self.stdout_file = open(os.path.join(self.output_folder, self.stdout_file_name), 'w')
        self.stdout_backup = sys.stdout
        sys.stdout = Tee(sys.stdout, self.stdout_file)

    def undirect_stdout(self):
        """
        Redirect stdout to default stdout.
        """
        if self.stdout_backup is not None:
            sys.stdout = self.stdout_backup
            self.stdout_backup = None
        if self.stdout_file is not None:
            self.stdout_file.close()
            self.stdout_file = None

    def copy_files(self):
        """
        Copy files to the output_folder.
        """
        if self.files_to_copy is not None:
            all_files_to_copy = []
            for file_to_copy in self.files_to_copy:
                all_files_to_copy += glob(file_to_copy)
            copy_files_to_folder(all_files_to_copy, self.output_folder)

    def folder_timestamp(self):
        """
        Return a timestamp as a folder name.
        :return: Current timestamp as string.
        """
        return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    def folder_base(self):
        """
        Return the base folder.
        :return: Base folder.
        """
        return self.output_folder

    def path(self, *paths):
        """
        Return the base folder appended with the given filename.
        :param paths: Paths to append (see os.path.join).
        :return: The folder.
        """
        if self.current_output_folder:
            return os.path.join(self.output_folder, self.current_output_folder, *paths)
        else:
            return os.path.join(self.output_folder, *paths)

    def path_for_iteration(self, iteration, *paths):
        """
        Return the base folder plus 'iter_{iteration}/*paths'.
        :param iteration: The current iteration.
        :param paths: Paths to append (see os.path.join).
        :return: The folder.
        """
        return self.path('iter_' + str(iteration), *paths)

    def set_current_path(self, *paths):
        """
        Set the current path for the iteration.
        :param paths: Paths to set (see os.path.join).
        """
        if len(paths) == 0 or paths[0] == None:
            self.current_output_folder = None
        else:
            self.current_output_folder = os.path.join(*paths)

    def set_current_path_for_iteration(self, iteration, *paths):
        """
        Set the current path for the iteration.
        :param iteration: The current iteration.
        :param paths: Paths to append (see os.path.join).
        """
        self.current_output_folder = os.path.join('iter_' + str(iteration), *paths)

    def folder(self, folder):
        """
        Return the base folder appended with the given folder.
        :param folder: Folder to append.
        :return: The folder.
        """
        print('DeprecationWarning: this function may removed in newer versions. Use path() instead.')
        return self.path(folder)

    def folder_for_iteration(self, iteration):
        """
        Return the base folder plus 'iter_{iteration}'.
        :param iteration: The current iteration.
        :return: The folder.
        """
        print('DeprecationWarning: this function may removed in newer versions. Use path() instead.')
        return self.path_for_iteration(iteration)

    def file(self, file_name):
        """
        Return the base folder appended with the given filename.
        :param file_name: Filename to append.
        :return: The folder.
        """
        print('DeprecationWarning: this function may removed in newer versions. Use path() instead.')
        return self.path(file_name)

    def file_for_iteration(self, file_name, iteration):
        """
        Return the base folder plus 'iter_{iteration}/file_name'.
        :param file_name: Filename to append.
        :param iteration: The current iteration.
        :return: The folder.
        """
        print('DeprecationWarning: this function may removed in newer versions. Use path_for_iteration() instead.')
        return self.path_for_iteration(iteration, file_name)
