# HoloLingo

Welcome to HoloLingo! This repository is where you'll discover an API designed for real-time translation coupled with speaker localization, tailored specifically for Mixed Reality applications. Building on top of an earlier [app](https://github.com/MixedRealityETHZ/Automatic-Transcription-and-Translation), which uses OpenAI's Whisper model for seamless multi-lingual translations, the latest extension adds the capability of localizing multiple speakers and overlaying translated text directly atop each speaker's location.

## The App

Current speaker localization methods typically rely on either video or audio input to identify the speaker. However, utilizing video can be expensive as it involves recording frames on a Mixed Reality device and transmitting them to an API for translation and speaker localization, resulting in significant delays and costs.

An alternative approach proposed by this API involves utilizing a single frame of the video along with stereo audio to streamline the process. By using Fast Fourier Transform, we can calculate the delay between microphones to perform Time Difference of Arrival (TDoA) analysis, determining the direction of sound. Then, by conducting people detection on the received image, we can identify potential speaker locations. This combined approach, incorporating both direction detection and image analysis, offers a more efficient and accurate approximation of the speaker's location.

How does this combination work?

Very simple. We begin by analyzing the received image with YOLOv8 to identify potential speakers. Based on the number of speakers detected, we partition the image into corresponding subsections. For instance, if there are three individuals in the image, we divide it into three sections: left, center, and right. Then, we use Fast Fourier Transform for sound localization, fine-tuning the delay threshold to differentiate between directions based on the subsections obtained. With both the direction and the speaker positions determined, we check which speaker is situated within that subsection of the frame, using the center of the bounding box to approximate their location.

Currently, we have the following subsections:

- 0 or 1 person: no subsections, with text displayed in the middle.
- 2 people: left and right.
- 3 people: left, center, and right.
- More than 4 people: left, front left, front right, and right.

The subsection division is currently limited to a number of 4 people since after some experiments, that typical usage scenarios with ML2 glasses involve conversational settings that capture no more than three or four individuals inside the frame.

This application samples audio every three seconds for translation, capturing an image at the onset of each three-second interval. However, this sampling frequency poses a limitation: the localization accuracy may suffer if the user moves extensively during these intervals.

## Performance and Limitations

For reference, the project was developed on a system with an NVIDIA GeForce GTX 1650 Ti graphics card, an Intel i7 processor, and 32GB of RAM.

Regarding performance, the current application runs speaker localization in approximately half a second and translation in 3.5 seconds using Faster Whisper. However, performance with the ML2 is less optimal due to the dependency on WiFi bandwidth for transmitting both audio and images, as well as the limitations of the ML2 hardware.

In terms of limitations, bandwidth is a critical factor since the overall experience of the application heavily depends on the time it takes for subtitles to appear. Additionally, the app is sensitive to noise. Currently, the app supports a maximum of four people and in this scenario, the app is sensitive to noise; slight noise on the opposite side of the speaker could cause subtitles to shift to the adjacent person.

Lastly, a demo video cannot be provided due to the limitations of the Magic Leap 2, as it cannot simultaneously run video capture and image capture for the Mixed Reality camera.

## Future Improvements

Future improvements for the app include the following:

- Implementing noise reduction to address sensitivity to noise in scenarios with around four people.
- Optimizing Whisper's translation to reduce processing time.
- Enhancing the Unity app for the Magic Leap 2.
- Developing apps for other Mixed Reality devices, such as the HoloLens, to explore performance on these platforms.

## Installation and Setup

### Prerequisites

- Magic Leap 2 glasses
- A server for running the HoloLingo API with CUDA

In case you want to modify some aspects of the project, you will also need:

- Windows (for ML2 development)
- Magic Leap Hub
- Unity v2022.3.10f1 (it is not guaranteed to work if another version is used)

### Environment

This project uses Poetry as dependency manager and packaging. In case you do not have install it, you can follor the instructions [here](https://python-poetry.org/docs/). After installing Poetry, you would need to create first a virtual environment with the following commands:

```bash
python -m venv .venv
. .\venv\Scripts\activate # If using Windows Powershell
. ./venv/bin/activate # If using Unix
```

Now, install the environment along with the HoloLingo package:

```bash
poetry install
```

#### Error: CUDA unavailable

To fix this, uninstall all `torch` libraries and reinstall them with this command:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
```

### Unity Project

First, create an empty Unity project and set it up to work with Magic Leap by following these [tutorials](https://developer-docs.magicleap.cloud/docs/guides/unity/getting-started/unity-getting-started/): "Create a Project", "Configure Project Settings", and "Render Pipeline Settings".

Once the project is set up with ML2, proceed to install MRTK3 in the project. Follow this [tutorial](https://developer-docs.magicleap.cloud/docs/guides/third-party/mrtk3/mrtk3-new-project/) and when prompted to select MRTK3 components for installation, choose "MRTK Input", "MRTK UX Components", and "MRTK UX Components (Non-canvas)". Make sure you check the `RECORD_AUDIO` permission in the final step!

To import the ATT project, go to `Assets -> Import Package -> Custom Package`, and then import the project package `hololingo.unitypackage`. Once imported, a prompt will appear asking you to `Import TMP essentials`. Click on it to finish Unity's setup.

Currently, the server's IP is hardcoded in the Unity app. Because of this, you'll need to change the IP to your own and rebuild the app. To change the IP to yours, navigate inside the Unity project to the `TranslationScene` and click on the `API` object. Change the IP in the inspector field with the IP (refer to image). Same in the `MainMenu` scene and the `AppStart` object.

![Unity Inspector showing IP field](/docs/images/unity_inspector.png)

Finally, build the Unity app and install it in the glasses using the MagicLeap Hub. Now that the project is completely setup!

## Usage

### HoloLingo API

To start the HoloLingo API, you need to activate the environment previously created and just execute:

```bash
python hololingo
```

You can specify the Whisper and YOLOv8 models you want to use with the options `--whisper` and `--yolo`. Currently, the defaults are `large-v3` for Faster Whisper, as it provides the best quality translations, and `yolov8m.pt` for YOLOv8, as it is both efficient and fast enough for people detection.

For more options, you can run the command with the `--help` option, which will display the available choices:

```bash
python hololingo --help
```

### Demo Videos

To process standard videos recorded with stereo audio, such as those from phones, you can create demo videos that showcase speaker localization and translation using the demo script.

First, extract the audio using the following command:

```bash
python demo/extract_audio.py <file_name>
```

Then, you can use the demo script as follows to obtain the videos:

```bash
python demo/demo.py <file_name>
```

In this script, you can specify the video, audio, and output folders with the options `--video`, `--audio`, and `--output`. For more information and additional options, use `--help`.

Please note that the app may have certain limitations and performance considerations, as mentioned in the previous sections.

## Acknowledgments

I would like to give a huge thanks to my friends and flatmates, as I couldn't have developed this project without their support. I deeply appreciate your patience with my continuous requests to record videos until the very end :D
