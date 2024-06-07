import logging
import os

import typer
from moviepy.editor import VideoFileClip
from rich.logging import RichHandler

FORMAT = "%(message)s"
logger = logging.getLogger(__name__)


def main(filename: str, video_dir: str = "videos", audio_dir: str = "audios") -> None:
    # Define the input video file and output audio file
    video_file = os.path.join(video_dir, filename + ".mp4")
    audio_file = os.path.join(audio_dir, filename + ".mp3")
    logger.info(f"Extracting audio from video file: {video_file}")
    logger.info(f"Saving audio to file: {audio_file}")

    # Load the video clip
    video_clip = VideoFileClip(video_file)

    # Extract the audio from the video clip
    audio_clip = video_clip.audio

    # Write the audio to a separate file
    audio_clip.write_audiofile(audio_file)

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    logger.info("Audio extraction successful!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%d-%m-%y %H:%M:%S %f]", handlers=[RichHandler(markup=True)])
    typer.run(main)
