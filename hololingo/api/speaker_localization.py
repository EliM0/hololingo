import logging
from enum import Enum
from typing import BinaryIO, List

import cv2
import numpy as np
import soundfile as sf
from ultralytics import YOLO

FORMAT = "%(message)s"
logger = logging.getLogger("uvicorn.error")

MODEL_CONFIDENCE = 0.8  # confidence threshold for YOLO model

DELAY_THRESHOLD_3_PEOPLE = 7.5e-06
DELAY_THRESHOLD_4_PEOPLE = 1.5e-05


# Map sound direction to delay
class Direction(Enum):
    FRONT = 0
    FRONT_RIGHT = 1
    FRONT_LEFT = 2
    RIGHT = 3
    LEFT = 4


# Localize sound direction
def _localize_sound(left_audio: np.ndarray, right_audio: np.ndarray, n_people: int) -> Direction:
    # Normalize signals
    left_signal = left_audio / np.max(np.abs(left_audio))
    right_signal = right_audio / np.max(np.abs(right_audio))

    # Estimate time delay between two signals using cross-correlation with Fast-Fourier Transform
    n = len(left_signal) + len(right_signal) - 1
    left_fft = np.fft.fft(left_signal, n=n)
    right_fft = np.fft.fft(right_signal[::-1], n=n)  # reverse for correlation
    correlation = np.fft.ifft(left_fft * right_fft).real

    delay_index = np.argmax(correlation) - (len(right_signal) - 1)
    delay = delay_index / len(right_signal)
    logger.info(f"Signal delay: {delay}")

    # Determine direction
    if n_people == 2:
        return Direction.LEFT if delay < 0 else Direction.RIGHT
    elif n_people == 3:
        if delay > DELAY_THRESHOLD_3_PEOPLE:
            return Direction.RIGHT
        elif delay < -DELAY_THRESHOLD_3_PEOPLE:
            return Direction.LEFT
        else:
            return Direction.FRONT
    else:
        if delay >= 0:
            return Direction.FRONT_RIGHT if delay < DELAY_THRESHOLD_4_PEOPLE else Direction.RIGHT
        else:
            return Direction.FRONT_LEFT if delay > -DELAY_THRESHOLD_4_PEOPLE else Direction.LEFT


# Determines if the person at that position is talking
def _isTalking(sound_dir: Direction, position: tuple, frame_section_width: int, n_people: int):
    return (
        (sound_dir == Direction.LEFT and position[0] < frame_section_width)
        or (sound_dir == Direction.RIGHT and position[0] > (n_people - 1) * frame_section_width)
        or (sound_dir == Direction.FRONT and position[0] > frame_section_width and position[0] < (n_people - 1) * frame_section_width)
        or (sound_dir == Direction.FRONT_RIGHT and position[0] > frame_section_width and position[0] < 2 * frame_section_width)
        or (sound_dir == Direction.FRONT_LEFT and position[0] > 2 * frame_section_width and position[0] < 3 * frame_section_width)
    )


def localize_speaker(audio_file: BinaryIO, frame_file: BinaryIO, model: YOLO) -> List:
    talking_coord = [-1, -1]

    try:
        frame = cv2.imread(frame_file)
        results = model.predict(frame, classes=[0], conf=MODEL_CONFIDENCE, verbose=False)  # Detect people in the frame
        frame_width = frame.shape[1]

        # Divide frame into sections based on number of people
        n_people = min(len(results[0].boxes.xyxy), 4)  # Limit to 4 people
        logger.info(f"Number of people (max. 4): {n_people}")

        # If no people detected or there is just one person, write text in default position
        if n_people <= 1:
            return talking_coord

        frame_section_width = frame_width // n_people

        # Process sound direction and time delay
        audio_data, _ = sf.read(audio_file)
        left_signal = audio_data[:, 0]
        right_signal = audio_data[:, 1]

        sound_dir = _localize_sound(left_signal, right_signal, n_people)
        logger.info(f"Sound direction: {sound_dir.name}")

        # Localize speaker
        for i, box in enumerate(results[0].boxes.xyxy):
            # Calculate box center
            x = int(box[0] + (box[2] - box[0]) / 2)
            y = int(box[1] + (box[3] - box[1]) / 2)

            # Check if that person is talking
            if _isTalking(sound_dir, (x, y), frame_section_width, n_people):
                talking_coord = [x, y]
    except:
        logger.exception("Exception during localizing speaker.")

    return talking_coord
