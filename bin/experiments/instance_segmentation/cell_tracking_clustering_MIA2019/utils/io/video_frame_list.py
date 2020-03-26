
import utils.io.text
from collections import OrderedDict
from utils.random import int_uniform, bool_bernoulli


class VideoFrameList(object):
    """
    This class loads and preprocesses a video frame list. Returns a list of IDs of neighboring frames
    for a given ID and parameters.
    """
    def __init__(self,
                 video_frame_list_file_name,
                 frames_before=None,
                 frames_after=None,
                 border_mode='duplicate',
                 random_start=False,
                 random_skip_probability=0.0):
        """
        Initializer.
        :param video_frame_list_file_name: The frames.csv file. Every lin represents a video, while the first
                                           column is the video id and the following entries are the ordered frames.
        :param frames_before: The number of frames before the given frame ID that should be considered in the resulting list.
        :param frames_after: The number of frames after the given frame ID that should be considered in the resulting list.
        :param border_mode: The border mode for frames outside the video.
        :param random_start: If true, the reference frame is not the given ID, but a random ID within the current range.
        :param random_skip_probability: The probability that a frame is being skipped.
        """
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
        """
        Load the frame list file.
        """
        self.video_id_frames = utils.io.text.load_dict_csv(self.video_frame_list_file_name)

    def get_frame_index_list(self, index, video_frames):
        """
        Return the current frame index list for the given index and video frames.
        :param index: The current frame index.
        :param video_frames: The list of video frames.
        :return: The current frame index list.
        """
        num_frames = self.frames_before + self.frames_after + 1
        if self.random_start:
            # shift start index, if random_start == True
            start_index = index - self.frames_before + int_uniform(0, num_frames - 1)
        else:
            start_index = index - self.frames_before
        index_list = []
        current_index = start_index
        # append the index_list
        while len(index_list) < num_frames:
            if self.random_skip_probability > 0 and bool_bernoulli(self.random_skip_probability) is True:
                # skip current index with a certain probability
                current_index += 1
                continue
            index_list.append(current_index)
            current_index += 1
        # if border mode is set to valid, shift all frames in case they are outside the video
        if self.border_mode == 'valid':
            if index_list[0] < 0:
                # shift to beginning
                shift = -index_list[0]
                index_list = [i + shift for i in index_list]
            elif index_list[-1] >= len(video_frames):
                # shift to end
                shift = -(index_list[-1] - len(video_frames) + 1)
                index_list = [i + shift for i in index_list]
        return index_list

    def get_video_frame_range_check(self, frame_index, video_frames):
        """
        Returns the current video frame ID and assert if the frame index is not within the current video frames.
        :param frame_index: The index to check.
        :param video_frames: The list of all video frames.
        :return: The current video frame ID.
        """
        assert frame_index >= 0 and frame_index < len(video_frames), 'invalid frame index'
        return video_frames[frame_index]

    def get_video_frame_duplicate(self, frame_index, video_frames):
        """
        Return a video frame, in duplicate border mode.
        :param frame_index: The index to check.
        :param video_frames: The list of all video frames.
        :return: The current video frame ID.
        """
        if frame_index < 0:
            frame_index = 0
        elif frame_index >= len(video_frames):
            frame_index = len(video_frames) - 1
        return self.get_video_frame_range_check(frame_index, video_frames)

    def get_video_frame_repeat(self, frame_index, video_frames):
        """
        Return a video frame, in repeate border mode.
        :param frame_index: The index to check.
        :param video_frames: The list of all video frames.
        :return: The current video frame ID.
        """
        while frame_index < 0:
            frame_index += len(video_frames)
        while frame_index >= len(video_frames):
            frame_index -= len(video_frames)
        return self.get_video_frame_range_check(frame_index, video_frames)

    def get_video_frame_mirror(self, frame_index, video_frames):
        """
        Return a video frame, in mirror border mode.
        :param frame_index: The index to check.
        :param video_frames: The list of all video frames.
        :return: The current video frame ID.
        """
        if frame_index < 0:
            frame_index = -frame_index
        elif frame_index >= len(video_frames):
            frame_index = len(video_frames) - (frame_index - len(video_frames)) - 1
        return self.get_video_frame_range_check(frame_index, video_frames)

    def get_video_frame(self, frame_index, video_frames):
        """
        Return a video frame. Handles different border modes.
        :param frame_index: The index to check.
        :param video_frames: The list of all video frames.
        :return: The current video frame ID.
        """
        if self.border_mode == 'duplicate':
            return self.get_video_frame_duplicate(frame_index, video_frames)
        elif self.border_mode == 'repeat':
            return self.get_video_frame_repeat(frame_index, video_frames)
        elif self.border_mode == 'mirror':
            return self.get_video_frame_mirror(frame_index, video_frames)
        else:
            return self.get_video_frame_range_check(frame_index, video_frames)

    def get_frame_ids(self, video_id, frame_id):
        """
        Return the current frame IDs for the given video and frame ID and the set parameters.
        :param video_id: The video ID.
        :param frame_id: The frame ID.
        :return: A list of frame IDs.
        """
        video_frames = self.video_id_frames[video_id]
        index = video_frames.index(frame_id)
        frame_index_list = self.get_frame_index_list(index, video_frames)

        frame_list = []
        for frame_index in frame_index_list:
            video_frame = self.get_video_frame(frame_index, video_frames)
            frame_list.append(video_frame)
        return frame_list

    def get_id_dict_list(self, video_id, frame_id):
        """
        Return the current video and frame id tuples for the given parameters.
        :param video_id: The video ID.
        :param frame_id: The frame ID.
        :return: A list of video ID frame ID tuples.
        """
        frame_ids = self.get_frame_ids(video_id, frame_id)
        return [OrderedDict([('video_id', video_id), ('frame_id', current_frame_id)]) for current_frame_id in frame_ids]
