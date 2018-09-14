
import utils.io.text
from collections import OrderedDict
from utils.random import int_uniform, bool_bernoulli

class VideoFrameList(object):
    def __init__(self,
                 video_frame_list_file_name,
                 frames_before=None,
                 frames_after=None,
                 border_mode='duplicate',
                 random_start=False,
                 random_skip_probability=0.0):
        self.video_frame_list_file_name = video_frame_list_file_name
        assert frames_before >= 0 and frames_after >= 0, 'number of frames must not be negative'
        self.frames_before = frames_before
        self.frames_after = frames_after
        assert border_mode in ['duplicate', 'repeat', 'mirror', 'valid'], 'invalid border mode'
        self.border_mode = border_mode
        self.random_start = random_start
        self.random_skip_probability = random_skip_probability
        self.video_id_frames = {}
        self.load()

    def load(self):
        self.video_id_frames = utils.io.text.load_dict_csv(self.video_frame_list_file_name)

    def get_frame_index_list(self, index, video_frames):
        num_frames = self.frames_before + self.frames_after + 1
        if self.random_start:
            start_index = index - int_uniform(0, num_frames)
        else:
            start_index = index - self.frames_before
        index_list = []
        current_index = start_index
        while len(index_list) < num_frames:
            if self.random_skip_probability > 0 and bool_bernoulli(self.random_skip_probability) is True:
                current_index += 1
                continue
            index_list.append(current_index)
            current_index += 1
        if self.border_mode == 'valid':
            if index_list[0] < 0:
                shift = -index_list[0]
                index_list = [i + shift for i in index_list]
            elif index_list[-1] >= len(video_frames):
                shift = -(index_list[-1] - len(video_frames) + 1)
                index_list = [i + shift for i in index_list]
        return index_list

    def get_video_frame_range_check(self, frame_index, video_frames):
        assert frame_index >= 0 and frame_index < len(video_frames), 'invalid frame index'
        return video_frames[frame_index]

    def get_video_frame_duplicate(self, frame_index, video_frames):
        if frame_index < 0:
            frame_index = 0
        elif frame_index >= len(video_frames):
            frame_index = len(video_frames) - 1
        return self.get_video_frame_range_check(frame_index, video_frames)

    def get_video_frame_repeat(self, frame_index, video_frames):
        while frame_index < 0:
            frame_index += len(video_frames)
        while frame_index >= len(video_frames):
            frame_index -= len(video_frames)
        return self.get_video_frame_range_check(frame_index, video_frames)

    def get_video_frame_mirror(self, frame_index, video_frames):
        if frame_index < 0:
            frame_index = -frame_index
        elif frame_index >= len(video_frames):
            frame_index = len(video_frames) - (frame_index - len(video_frames)) - 1
        return self.get_video_frame_range_check(frame_index, video_frames)

    def get_video_frame(self, frame_index, video_frames):
        if self.border_mode == 'duplicate':
            return self.get_video_frame_duplicate(frame_index, video_frames)
        elif self.border_mode == 'repeat':
            return self.get_video_frame_repeat(frame_index, video_frames)
        elif self.border_mode == 'mirror':
            return self.get_video_frame_mirror(frame_index, video_frames)
        else:
            return self.get_video_frame_range_check(frame_index, video_frames)

    def get_image_ids(self, video_id, frame_id):
        video_frames = self.video_id_frames[video_id]
        index = video_frames.index(frame_id)
        frame_index_list = self.get_frame_index_list(index, video_frames)

        frame_list = []
        for frame_index in frame_index_list:
            video_frame = self.get_video_frame(frame_index, video_frames)
            frame_list.append(video_frame)
        return frame_list

    def get_id_dict_list(self, video_id, frame_id):
        frame_ids = self.get_image_ids(video_id, frame_id)
        return [OrderedDict([('video_id', video_id), ('frame_id', current_frame_id)]) for current_frame_id in frame_ids]
