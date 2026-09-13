import math

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

video_path = "data/test_videos/test_video.mp4"
model_path = "models/pose_landmarker_lite.task"
window_name = "SafeWatch - Day 9 Posture Features"

process_every_n_frames = 3

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

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, target_width, target_height)

frame_number = 0
last_landmarks = None

print("Day 9 started.")
print("Press 'q' to close the video.")

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

        posture = "UNKNOWN"
        posture_color = (0, 165, 255)

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

            if torso_angle > 60:
                posture = "UPRIGHT"
                posture_color = (0, 255, 0)
            elif torso_angle < 35:
                posture = "HORIZONTAL"
                posture_color = (0, 0, 255)

            cv2.line(
                resized_frame,
                shoulder_midpoint,
                hip_midpoint,
                (255, 0, 255),
                3,
            )

            cv2.circle(resized_frame, shoulder_midpoint, 6, (255, 0, 255), -1)
            cv2.circle(resized_frame, hip_midpoint, 6, (255, 0, 255), -1)

            cv2.putText(
                resized_frame,
                f"Angle: {torso_angle:.1f} deg",
                (15, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.putText(
                resized_frame,
                f"Body W: {body_width}px | H: {body_height}px",
                (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        cv2.putText(
            resized_frame,
            f"Posture: {posture}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            posture_color,
            2,
        )

        cv2.putText(
            resized_frame,
            "Purple line: shoulder midpoint to hip midpoint",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            resized_frame,
            "Q: Quit",
            (15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.imshow(window_name, resized_frame)

        frame_number += 1

        if cv2.waitKey(30) & 0xFF == ord("q"):
            print("Video closed by user.")
            break

video.release()
cv2.destroyAllWindows()