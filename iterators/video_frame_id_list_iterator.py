
from iterators.iterator_base import IteratorBase
import utils.io.text
import random
import multiprocessing
import os


class VideoFrameIdListIterator(IteratorBase):
    def __init__(self,
                 id_list_file_name,
                 video_frame_list,
                 random=False,
                 keys=None,
                 num_frames=None,
                 border_mode='duplicate',
                 random_start=False,
                 skip_probability=0.0):
        self.id_list_file_name = id_list_file_name
        self.video_frame_list = video_frame_list
        self.random = random
        self.keys = keys
        if self.keys is None:
            self.keys = ['image_id']
        self.num_frames = num_frames
        self.border_mode = border_mode
        self.random_start = random_start
        self.skip_probability = skip_probability
        self.lock = multiprocessing.Lock()
        self.load()
        self.reset()

    def load(self):
        ext = os.path.splitext(self.id_list_file_name)[1]
        if ext in ['.csv', '.txt']:
            self.id_list = utils.io.text.load_list_csv(self.id_list_file_name)
        print('loaded %i ids' % len(self.id_list))

    def reset(self):
        self.index_list = list(range(len(self.id_list)))
        if self.random:
            random.shuffle(self.index_list)
        self.current_index = 0

    def num_entries(self):
        return len(self.id_list)

    def get_next_id(self):
        with self.lock:
            if self.current_index >= len(self.id_list):
                self.reset()
            current_id_list = self.id_list[self.index_list[self.current_index]]
            self.current_index += 1
            current_dict = dict(zip(self.keys, current_id_list))
            #current_dict['unique_id'] = '_'.join(map(str, current_id_list))
            return self.video_frame_list.get_id_dict_list(**current_dict)
