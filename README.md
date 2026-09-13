# SafeWatch: Pose-Based Possible Fall Detection

SafeWatch is a lightweight computer-vision prototype that analyzes a person’s body pose in a local video and flags a possible fall.

It uses MediaPipe Pose Landmarker to detect body landmarks, OpenCV to process video frames, and rule-based temporal logic to identify an upright-to-horizontal transition followed by sustained horizontal posture.

> This is an educational AIML/computer-vision project. It is not a medical device, emergency-response system, or certified fall detector.

---

## Features

- Processes local video files without requiring a webcam
- Detects 33 human body pose landmarks using MediaPipe
- Draws pose landmarks and body connections on the video
- Calculates torso angle and body width-to-height ratio
- Classifies posture as `UPRIGHT`, `HORIZONTAL`, or `UNKNOWN`
- Detects a possible fall using posture changes across multiple frames
- Shows a visible `WARNING: POSSIBLE FALL DETECTED` banner
- Saves an alert screenshot with timestamp
- Stores fall-event details in a CSV log
- Exports an annotated output video for demonstration
- Optimized for a lower-powered laptop by resizing frames and processing pose estimation every third frame

---

## How It Works

```text
Local video file
        ↓
OpenCV reads and resizes video frames
        ↓
MediaPipe Pose Landmarker detects body landmarks
        ↓
Torso angle + body ratio are calculated
        ↓
Posture is classified over multiple frames
        ↓
Upright → horizontal transition is detected
        ↓
Possible-fall warning, alert screenshot, CSV event log, and processed video
```

### Detection logic

The program flags a possible fall only when it observes:

1. An `UPRIGHT` posture recently
2. A transition to `HORIZONTAL`
3. Sustained horizontal posture for multiple observations

This helps reduce false alerts from a single unusual frame or a person who is already lying down.

---

## Tech Stack

- Python 3.14
- OpenCV
- MediaPipe 1.0.1
- NumPy
- CSV module from the Python standard library

---

## Project Structure

```text
SafeWatch-Fall-Detection/
│
├── data/
│   └── test_videos/
│       ├── test_video.mp4
│       └── fall_test_video.mp4
│
├── models/
│   └── pose_landmarker_lite.task
│
├── outputs/
│   ├── alerts/
│   ├── processed_videos/
│   ├── screenshots/
│   └── events.csv
│
├── day1_test.py
├── day2_webcam_test.py
├── day3_video_screenshot.py
├── day4_video_info.py
├── day6_pose_image.py
├── day7_pose_video.py
├── day8_optimized_pose_video.py
├── day9_posture_features.py
├── day10_stable_posture.py
├── fall_detection.py
├── save_processed_video.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone [https://github.com/https://github.com/Shreya-253/SafeWatch-Fall-Detection.git](https://github.com/YOUR-GITHUB-USERNAME/SafeWatch-Fall-Detection.git)
cd SafeWatch-Fall-Detection
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Download the MediaPipe pose model

Download the MediaPipe Pose Landmarker Lite model:

https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Create a folder named `models` if it does not exist, then place the downloaded file here:

```text
models/pose_landmarker_lite.task
```

### 4. Add your local test videos

Place video files inside:

```text
data/test_videos/
```

For the current program configuration, use:

```text
data/test_videos/fall_test_video.mp4
```

---

## Run the Project

### Interactive fall-detection demo

```bash
python fall_detection.py
```

The program opens the video in an OpenCV window and displays pose landmarks, posture status, and a possible-fall warning.

Press:

```text
Q = Quit
```

### Save an annotated output video

```bash
python save_processed_video.py
```

The processed video is saved here:

```text
outputs/processed_videos/safeWatch_fall_demo.mp4
```

The program also saves:

```text
outputs/alerts/
outputs/events.csv
```

---

## Testing

The prototype was tested on two local video scenarios:

| Test video | Expected behavior | Observed result |
|---|---|---|
| Normal walking video | No alert | No possible-fall alert triggered |
| Staged indoor fall-like video | One possible-fall alert | Alert banner, screenshot, event-log entry, and annotated video created |

This project has been tested as a small prototype. No formal dataset-wide accuracy, precision, recall, or medical-safety claim is made.

---

## Performance Optimization

The original test video had high resolution, so the program was optimized to run on a lower-powered laptop:

- Frames are resized before pose detection
- Pose estimation runs every third frame
- Previous landmarks are reused between processed frames
- The MediaPipe Lite pose model is used

---

## Limitations

- This is not a medical or emergency-use application.
- It detects a possible fall based on visible body posture and movement.
- Results can be affected by camera angle, poor lighting, motion blur, partial body visibility, multiple people, or occlusion.
- A person lying down intentionally may be mistaken for a fall in some situations.
- The current version is designed for one visible person at a time.
- More testing with diverse staged videos would be needed before making performance claims.

---

## Video Credit

Testing used locally downloaded, publicly available stock footage from Pexels under the Pexels License.

The original videos are not included in this repository.

---

## Learning Journey

This project was built as a short hands-on learning project to understand:

- Python project structure
- Git and GitHub Desktop
- OpenCV video processing
- Video frames, FPS, and resolution
- MediaPipe pose landmarks
- Landmark feature extraction
- Rule-based posture classification
- Temporal event detection
- Logging and annotated-video generation

---

## Future Improvements

- Test on more staged fall and non-fall videos
- Add better posture features such as hip velocity and vertical displacement
- Add configurable detection thresholds
- Support multiple-person detection
- Train a temporal classifier such as an LSTM or GRU
- Build a lightweight user interface in the future
