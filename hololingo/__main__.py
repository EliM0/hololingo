import os
from enum import Enum

import typer
import uvicorn
from faster_whisper import WhisperModel


class WhisperModel(str, Enum):
    tiny = "tiny"
    small = "small"
    medium = "medium"
    large = "large"
    large_v2 = "large-v2"
    large_v3 = "large-v3"


class YOLOModel(str, Enum):
    yolov8n = "yolov8n.pt"
    yolov8s = "yolov8s.pt"
    yolov8m = "yolov8m.pt"
    yolov8l = "yolov8l.pt"
    yolov8x = "yolov8x.pt"


def main(
    host: str = typer.Argument(default="0.0.0.0", help="Host to run the server on."),
    port: int = typer.Argument(default=5000, help="Port to run the server on."),
    whisper: WhisperModel = typer.Argument(default=WhisperModel.large_v3, help="Whisper model to use."),
    yolo: YOLOModel = typer.Argument(default=YOLOModel.yolov8m, help="YOLO model to use."),
    debug: bool = typer.Option(False, help="Log level debug."),
) -> None:
    os.environ["WHISPER_MODEL"] = whisper.value
    os.environ["YOLO_MODEL"] = yolo.value

    log_level = "debug" if debug else "info"

    uvicorn.run("hololingo.api.api:app", host=host, port=port, reload=True, log_level=log_level, lifespan="on")


if __name__ == "__main__":
    typer.run(main)
