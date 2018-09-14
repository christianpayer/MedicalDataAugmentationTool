
import os
import csv
import re
import numpy as np
from utils.landmark.common import Landmark
from utils.io.common import create_directories_for_file_name

def load(file_name, num_landmarks, dim):
    ext = os.path.splitext(file_name)[1]
    if ext == '.csv':
        return load_csv(file_name, num_landmarks, dim)
    if ext == '.idl':
        return load_idl(file_name, num_landmarks, dim)
    else:
        raise RuntimeError(ext + ' filetype not supported')


def load_csv(file_name, num_landmarks, dim):
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            id = row[0]
            landmarks = []
            num_entries = dim * num_landmarks + 1
            assert num_entries == len(row), 'number of row entries and landmark coordinates do not match'
            # print(len(points_dict), name)
            for i in range(1, dim * num_landmarks + 1, dim):
                # print(i)
                if np.isnan(float(row[i])):
                    landmark = Landmark(None, False)
                else:
                    if dim == 2:
                        coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                    elif dim == 3:
                        coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    landmark = Landmark(coords)
                landmarks.append(landmark)
            landmarks_dict[id] = landmarks
    return landmarks_dict


def load_multi_csv(file_name, num_landmarks, dim=2):
    landmarks_dict = {}
    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name = row[0]
            if name in landmarks_dict:
                landmarks_dict_per_image = landmarks_dict[name]
            else:
                landmarks_dict_per_image = {}
                landmarks_dict[name] = landmarks_dict_per_image

            person_id = row[1]
            #if int(row[1]) != 0:
            #    continue
            landmarks = []
            #print(len(points_dict), name)
            for i in range(2, dim * num_landmarks + 2, dim):
                # print(i)
                if np.isnan(float(row[i])):
                    landmark = Landmark(None, False)
                else:
                    if dim == 2:
                        coords = np.array([float(row[i]), float(row[i + 1])], np.float32)
                    elif dim == 3:
                        coords = np.array([float(row[i]), float(row[i + 1]), float(row[i + 2])], np.float32)
                    landmark = Landmark(coords)
                landmarks.append(landmark)
            landmarks_dict_per_image[person_id] = landmarks
    return landmarks_dict


def load_idl(file_name, num_landmarks, dim):
    landmarks_dict = {}
    with open(file_name, 'r') as file:
        for line in file.readlines():
            id_match = re.search('"(.*)"', line)
            id = id_match.groups()[0]
            coords_matches = re.findall('\((\d*),(\d*),(\d*)\)', line)
            assert num_landmarks == len(coords_matches), 'number of row entries and landmark coordinates do not match'
            if dim == 2:
                landmarks = [Landmark(np.array([float(coords_match[0]), float(coords_match[1])], np.float32)) for coords_match in coords_matches]
            elif dim == 3:
                landmarks = [Landmark(np.array([float(coords_match[0]), float(coords_match[1]), float(coords_match[2])], np.float32)) for coords_match in coords_matches]
            landmarks_dict[id] = landmarks
    return landmarks_dict


def load_lml(file_name, num_landmarks, landmark_ids):
    landmarks = [Landmark() for _ in range(num_landmarks)]
    with open(file_name, 'r') as file:
        for line in file.readlines():
            if line.startswith('#'):
                continue
            tokens = line.split('\t')
            coords = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
            current_id = int(tokens[0])
            if current_id not in landmark_ids:
                print('Warning: invalid id {} for file {}'.format(current_id, file_name))
                continue
            landmark_index = landmark_ids.index(int(tokens[0]))
            landmarks[landmark_index] = Landmark(coords)
    return landmarks


def save_points_csv(landmarks_dict, filename, id_preprocessing=None):
    create_directories_for_file_name(filename)
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, values in sorted(landmarks_dict.items()):
            current_id = key
            if id_preprocessing is not None:
                current_id = id_preprocessing(current_id)
            row = [current_id]
            for landmark in values:
                row += list(landmark.coords)
            writer.writerow(row)


def save_points_idl(landmarks_dict, filename, id_preprocessing=None):
    create_directories_for_file_name(filename)
    with open(filename, 'w') as file:
        for key, values in sorted(landmarks_dict.items()):
            current_id = key
            if id_preprocessing is not None:
                current_id = id_preprocessing(current_id)
            string = '"' + current_id + '": '
            for landmark in values:
                point = landmark.coords
                if len(point) == 2:
                    string += '(' + '{:.3f}'.format(point[0]) + ',' + '{:.3f}'.format(point[1]) + ',0),'
                else:
                    string += '(' + '{:.3f}'.format(point[0]) + ',' + '{:.3f}'.format(point[1]) + ',' + '{:.3f}'.format(point[2]) + '),'

            string = string[:-1] + '\n'
            file.write(string)


def save_idl(data, file_name):
    with open(file_name, 'w') as file:
        for key, value in sorted(data.items()):
            string = '"' + key + '": '
            for point in value:
                if len(point) == 2:
                    string += '(' + '{:.3f}'.format(point[0]) + ',' + '{:.3f}'.format(point[1]) + ',0),'
                else:
                    string += '(' + '{:.3f}'.format(point[0]) + ',' + '{:.3f}'.format(point[1]) + ',' + '{:.3f}'.format(point[2]) + '),'

            string = string[:-1] + '\n'
            file.write(string)