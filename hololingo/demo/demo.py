import logging
import os
import textwrap
from typing import Tuple

import cv2
import requests
import soundfile as sf
import typer
from rich.logging import RichHandler

from hololingo.demo.utils import getAudioChunks, getVideoCaptures, save_audio_video

FORMAT = "%(message)s"
logger = logging.getLogger(__name__)
baseUrl = "http://localhost:5000"
tmp_dir = "./tmp"


def main(filename: str, video_dir: str = "videos", audio_dir: str = "audios", output_dir: str = "output", verbose: bool = False) -> None:
    if verbose:
        logger.setLevel(logging.DEBUG)

    # Open input video
    logger.info(f"Opening video file: {filename}")
    in_video_filename = os.path.join(video_dir, filename + ".mp4")
    tmp_video_filename = f"{tmp_dir}/output.mp4"
    input_video, output_video, fps = getVideoCaptures(in_video_filename, tmp_video_filename)
    if not input_video:
        exit(1)

    # Get audio chunks
    logger.info(f"Getting audio chunks from '{filename}'")
    audio_filename = os.path.join(audio_dir, filename + ".mp3")
    audio_chunks, samplerate = getAudioChunks(audio_filename)
    frames_to_skip = int(fps * 3)
    frame_count = 0

    frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_coords = (frame_width // 4, frame_height - 150)

    # Text settings
    font = cv2.FONT_HERSHEY_COMPLEX
    font_size = 1
    font_thickness = 2

    translation_text: str = ""
    speaker_coords: Tuple[int, int] = default_coords

    while True:
        ret, frame = input_video.read()

        # If there are no more frames, break the loop
        if not ret:
            break

        # Every 3 seconds, detect who's talking
        if frame_count % frames_to_skip == 0 and audio_chunks:
            files = {}
            # image_bytes = cv2.imencode(".jpg", frame)[1].tobytes()
            cv2.imwrite(f"{tmp_dir}/image.jpg", frame)
            sf.write(f"{tmp_dir}/audio.wav", data=audio_chunks.pop(0), samplerate=samplerate)

            with open(f"{tmp_dir}/audio.wav", "rb") as audio_buffer, open(f"{tmp_dir}/image.jpg", "rb") as image:
                files = {"audio": ("audio.wav", audio_buffer, "audio/wav"), "image": ("image.jpg", image, "image/jpeg")}

                # Send POST request with the image file
                logger.info(f"Sending request to {baseUrl}/translate")
                response = requests.post(baseUrl + "/translate", files=files)
                response_json = response.json()
                logger.info(f"Response: {response} {response_json}")

                speaker_coords = (int(response_json["speaker_coords"]["x"]), int(response_json["speaker_coords"]["y"]))

                translation_text = response_json["text"]
                wrapped_text = textwrap.wrap(translation_text, width=20)
                if speaker_coords == (-1, -1):
                    logger.info("No speaker detected")
                    speaker_coords = default_coords

        for i, line in enumerate(wrapped_text):
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

            gap = textsize[1] + 10

            y = int(speaker_coords[1] + (textsize[1]) / 2) + i * gap
            x = int(speaker_coords[0] - (textsize[0]) / 2)

            cv2.putText(frame, line, (x, y), font, font_size, (0, 255, 255), font_thickness, cv2.LINE_4)

        # Increment frame count
        frame_count += 1

        output_video.write(frame)

    # Release the videos
    logger.info("Releasing video captures")
    input_video.release()
    output_video.release()

    logger.info("Video processing complete")

    # Add audio to the output video
    save_audio_video(audio_filename, tmp_video_filename, output_dir, filename)

    logger.info("Finish!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%d-%m-%y %H:%M:%S %f]", handlers=[RichHandler(markup=True)])
    typer.run(main)
