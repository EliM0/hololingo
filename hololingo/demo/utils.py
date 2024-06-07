import logging
import os
from typing import List, Tuple, Union

import cv2
import soundfile as sf
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip

FORMAT = "%(message)s"
logger = logging.getLogger(__name__)


# Divides the audio in chunks of <chunk_duration> seconds
def getAudioChunks(audio_file: str, chunk_duration: int = 3) -> Tuple[List, int]:
    audio_data, sample_rate = sf.read(audio_file)

    chunk_nsamples = int(sample_rate * chunk_duration)
    num_chunks = len(audio_data) // chunk_nsamples

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_nsamples
        end = (i + 1) * chunk_nsamples
        chunks.append(audio_data[start:end])

    return chunks, sample_rate


# Opens input video and creates output video capture
def getVideoCaptures(in_video_filename: str, out_video_filename: str) -> Tuple[Union[cv2.VideoCapture, None], Union[cv2.VideoWriter, None], float]:
    # Load input video
    input_video = cv2.VideoCapture(in_video_filename)
    if not input_video.isOpened():
        logger.error(f"Couldn't open the video file: {in_video_filename}")
        return None, None, 0.0

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    fps = input_video.get(cv2.CAP_PROP_FPS)
    output_video = cv2.VideoWriter(out_video_filename, fourcc, fps, (int(input_video.get(3)), int(input_video.get(4))))

    return input_video, output_video, fps


# Combines audio and video files
def save_audio_video(audio_filename: str, tmp_video_filename: str, output_dir: str, filename: str) -> None:
    videoclip = VideoFileClip(tmp_video_filename)
    audioclip = AudioFileClip(audio_filename)

    new_audioclip = CompositeAudioClip([audioclip])
    videoclip.audio = new_audioclip
    out_video_filename = os.path.join(output_dir, filename + ".mp4")

    videoclip.write_videofile(out_video_filename)
    os.remove(tmp_video_filename)
