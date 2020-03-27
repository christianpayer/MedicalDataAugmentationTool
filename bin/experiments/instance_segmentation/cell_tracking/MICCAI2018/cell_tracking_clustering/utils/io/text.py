
import csv
from utils.io.common import create_directories_for_file_name

def load_dict_csv(file_name):
    d = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            value = row[1:]
            if len(value) == 1:
                value = value[0]
            d[id] = value
    return d


def load_list(file_name):
    with open(file_name, 'r') as file:
        return [line.strip('\n') for line in file.readlines()]


def load_list_csv(file_name):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def save_dict_csv(d, file_name, header=None, **kwargs):
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as file:
        writer = csv.writer(file, **kwargs)
        if header is not None:
            writer.writerow(header)
        for key, value in sorted(d.items()):
            if isinstance(value, list):
                writer.writerow([key] + value)
            elif isinstance(value, tuple):
                writer.writerow([key] + list(value))
            else:
                writer.writerow([key, value])


def save_list_csv(l, file_name, header=None, **kwargs):
    create_directories_for_file_name(file_name)
    with open(file_name, 'w') as file:
        writer = csv.writer(file, **kwargs)
        if header is not None:
            writer.writerow(header)
        for value in l:
            if isinstance(value, list):
                writer.writerow(value)
            elif isinstance(value, tuple):
                writer.writerow(list(value))
            else:
                writer.writerow([value])
