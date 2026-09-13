import csv
import math
import os
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

video_path = "data/test_videos/fall_test_video2.mp4"
model_path = "models/pose_landmarker_lite.task"

alerts_folder = "outputs/alerts"
events_file = "outputs/events.csv"
processed_videos_folder = "outputs/processed_videos"

window_name = "SafeWatch - Saving Processed Video"

process_every_n_frames = 3

history_size = 8
upright_needed = 5
horizontal_needed = 5

horizontal_after_transition_needed = 20
recent_upright_memory = 75
alert_cooldown_frames = 500

os.makedirs(alerts_folder, exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs(processed_videos_folder, exist_ok=True)

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"Error: Could not open video: {video_path}")
    raise SystemExit

original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    print("Error: Could not read FPS from the video.")
    video.release()
    raise SystemExit

if original_height > original_width:
    target_width = 360
    target_height = 640
else:
    target_width = 640
    target_height = 360

output_video_path = os.path.join(
    processed_videos_folder,
    "safeWatch_fall_demo.mp4",
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    output_video_path,
    fourcc,
    fps,
    (target_width, target_height),
)

if not video_writer.isOpened():
    print("Error: Could not create the output video file.")
    video.release()
    raise SystemExit

connections = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

if not os.path.exists(events_file):
    with open(events_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "event_time",
                "video_time_seconds",
                "frame_number",
                "event_type",
                "torso_angle",
                "body_ratio",
                "alert_image",
            ]
        )

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, target_width, target_height)

frame_number = 0
last_landmarks = None
posture_history = deque(maxlen=history_size)

stable_posture = "UNKNOWN"
last_upright_frame = -10000
horizontal_after_upright_count = 0

last_alert_frame = -10000
fall_detected_in_video = False

alert_frames_remaining = 0
alert_message = ""
alert_image_path = None

print("SafeWatch output-video generation started.")
print(f"Input: {video_path}")
print(f"Output: {output_video_path}")
print("Press 'q' to stop early.")

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = video.read()

        if not success:
            print("Video finished.")
            break

        resized_frame = cv2.resize(
            frame,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

        if frame_number % process_every_n_frames == 0:
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int((frame_number / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                last_landmarks = result.pose_landmarks[0]
            else:
                last_landmarks = None

        raw_posture = "UNKNOWN"
        torso_angle = 0.0
        body_ratio = 0.0

        if last_landmarks:
            points = []

            for landmark in last_landmarks:
                x = int(landmark.x * target_width)
                y = int(landmark.y * target_height)
                points.append((x, y))

            for start_index, end_index in connections:
                cv2.line(
                    resized_frame,
                    points[start_index],
                    points[end_index],
                    (0, 255, 0),
                    2,
                )

            for x, y in points:
                cv2.circle(resized_frame, (x, y), 3, (0, 255, 0), -1)

            left_shoulder = points[11]
            right_shoulder = points[12]
            left_hip = points[23]
            right_hip = points[24]

            shoulder_midpoint = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2,
            )

            hip_midpoint = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2,
            )

            torso_dx = hip_midpoint[0] - shoulder_midpoint[0]
            torso_dy = hip_midpoint[1] - shoulder_midpoint[1]

            torso_angle = abs(math.degrees(math.atan2(torso_dy, torso_dx)))

            body_width = max(x for x, y in points) - min(x for x, y in points)
            body_height = max(y for x, y in points) - min(y for x, y in points)

            if body_height > 0:
                body_ratio = body_width / body_height

            if torso_angle > 60 and body_ratio < 0.9:
                raw_posture = "UPRIGHT"
            elif torso_angle < 35 or body_ratio > 1.15:
                raw_posture = "HORIZONTAL"

            posture_history.append(raw_posture)

            if posture_history.count("UPRIGHT") >= upright_needed:
                stable_posture = "UPRIGHT"
            elif posture_history.count("HORIZONTAL") >= horizontal_needed:
                stable_posture = "HORIZONTAL"
            else:
                stable_posture = "UNKNOWN"

            cv2.line(
                resized_frame,
                shoulder_midpoint,
                hip_midpoint,
                (255, 0, 255),
                3,
            )

        if stable_posture == "UPRIGHT":
            last_upright_frame = frame_number
            horizontal_after_upright_count = 0
            posture_color = (0, 255, 0)

        elif stable_posture == "HORIZONTAL":
            posture_color = (0, 0, 255)

            if frame_number - last_upright_frame <= recent_upright_memory:
                horizontal_after_upright_count += 1
            else:
                horizontal_after_upright_count = 0

        else:
            posture_color = (0, 165, 255)
            horizontal_after_upright_count = 0

        possible_fall = (
            stable_posture == "HORIZONTAL"
            and horizontal_after_upright_count >= horizontal_after_transition_needed
            and frame_number - last_alert_frame >= alert_cooldown_frames
            and not fall_detected_in_video
        )

        if possible_fall:
            current_time_seconds = frame_number / fps
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            alert_message = "POSSIBLE FALL DETECTED"
            alert_frames_remaining = 180

            last_alert_frame = frame_number
            fall_detected_in_video = True

            alert_image_path = os.path.join(
                alerts_folder,
                f"possible_fall_{timestamp}.jpg",
            )

            with open(events_file, "a", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        f"{current_time_seconds:.2f}",
                        frame_number,
                        "POSSIBLE_FALL",
                        f"{torso_angle:.2f}",
                        f"{body_ratio:.2f}",
                        alert_image_path,
                    ]
                )

            print(
                f"Possible fall detected at {current_time_seconds:.2f} seconds."
            )

        cv2.putText(
            resized_frame,
            "SafeWatch: Pose-Based Fall Detection",
            (15, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            resized_frame,
            f"Posture: {stable_posture}",
            (15, 172),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            posture_color,
            2,
        )

        cv2.putText(
            resized_frame,
            f"Angle: {torso_angle:.1f} | Ratio: {body_ratio:.2f}",
            (15, 197),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            resized_frame,
            "Green: pose | Purple: torso",
            (15, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )

        if alert_frames_remaining > 0:
            cv2.rectangle(
                resized_frame,
                (0, 0),
                (target_width, 120),
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                resized_frame,
                "WARNING",
                (20, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                3,
            )

            cv2.putText(
                resized_frame,
                alert_message,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            alert_frames_remaining -= 1

        if possible_fall and alert_image_path:
            cv2.imwrite(alert_image_path, resized_frame)
            print(f"Alert screenshot saved: {alert_image_path}")

        video_writer.write(resized_frame)

        cv2.imshow(window_name, resized_frame)

        frame_number += 1

        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Stopped early by user.")
            break

video.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processed video saved successfully: {output_video_path}")