from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm


class Detector:

    def __init__(self):
        self.stats_manager = StatsManager()
        self.scene_manager = SceneManager(self.stats_manager)
        self.scene_manager.add_detector(ContentDetector())

    def _get_frames_timecodes(self, video_file_path):
        video_manager = VideoManager([video_file_path])
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        self.scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = self.scene_manager.get_scene_list(base_timecode)
        return scene_list

    def get_scene_frames(self, video_file_path):
        scene_list = self._get_frames_timecodes(video_file_path)
        video_manager = VideoManager([video_file_path])
        video_manager.start()
        for frame_timecode, _ in tqdm(scene_list):
            video_manager.seek(frame_timecode)
            _, frame = video_manager.read()
            yield frame
