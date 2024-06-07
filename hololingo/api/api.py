import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Union

import torch
from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel
from pydantic_settings import BaseSettings
from ultralytics import YOLO

from hololingo.api.speaker_localization import localize_speaker
from hololingo.api.utils import SUPPORTED_LANGUAGES

logger = logging.getLogger("uvicorn.error")


class Settings(BaseSettings):
    whisper: Union[WhisperModel, None] = None
    yolo: Union[YOLO, None] = None


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    on_gpu = torch.cuda.is_available()
    logger.info(f"Is CUDA available? {on_gpu}. If CUDA is not available, check installation instructions in README.")

    # Load the Whisper model
    model_name = os.getenv("WHISPER_MODEL")
    settings.whisper = WhisperModel(model_name, device="cuda", compute_type="auto")
    logger.info(f"Loaded Whisper model: {model_name}")

    # Load the YOLO model
    model_name = os.getenv("YOLO_MODEL")
    settings.yolo = YOLO(model_name)
    dummy_input = torch.zeros(1, 3, 640, 640)
    settings.yolo.predict(dummy_input, classes=[0], conf=0.8, verbose=False)  # Dummy prediction to load the model
    logger.info(f"Loaded YOLO model: {model_name}")

    yield

    logger.info("Shutting down server.")


app = FastAPI(title="HoloLingo API", description="HoloLingo API for translation with multiple speaker localization.", version="0.1.0", lifespan=lifespan)


@app.get("/")
def home():
    return {
        "message": "Welcome to the Translation Service API!",
        "endpoints": {
            "/translate": "POST endpoint to translate an audio file. Optionally provide 'from_language'.",
            "/transcribe": "POST endpoint to transcribe an audio file. Optionally provide 'from_language'.",
            "/supportedLanguages": "GET endpoint to list supported languages or check if a specific language is supported.",
        },
    }


@app.get("/supportedLanguages/{language}")
def supported_languages(language: str = None):
    if language:
        return {"supported": language in SUPPORTED_LANGUAGES}
    else:
        return {"languages": SUPPORTED_LANGUAGES}


@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    # Transcribe
    result, _ = settings.whisper.transcribe(audio.file, beam_size=5)
    text = " ".join(segment.text for segment in result)

    response = {"text": text}
    logger.info(f"Response: {response}")
    return response


@app.post("/translate")
async def translate(audio: UploadFile, image: UploadFile):
    # Find speaker
    start = time.time()
    speaker_coords = localize_speaker(audio.file, image.file, settings.yolo)
    end = time.time()
    logger.debug(f"Speaker localization took {end - start:.2f} seconds.")

    # Go back to the beginning of the audio file
    await audio.seek(0)

    # Translate
    start = time.time()
    result, _ = settings.whisper.transcribe(audio.file, beam_size=5, task="translate", without_timestamps=True)
    end = time.time()
    logger.debug(f"Translation took {end - start:.2f} seconds.")

    text = " ".join(segment.text for segment in result)
    response = {"text": text, "speaker_coords": {"x": speaker_coords[0], "y": speaker_coords[1]}}
    logger.info(f"Response: {response}")
    return response
