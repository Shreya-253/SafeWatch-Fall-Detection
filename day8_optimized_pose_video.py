import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

video_path = "data/test_videos/test_video.mp4"
model_path = "models/pose_landmarker_lite.task"
window_name = "SafeWatch - Day 8 Optimized Pose Video"

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

print(f"Original video: {original_width} x {original_height}")
print(f"Processed frame size: {target_width} x {target_height}")
print(f"Video FPS: {fps:.2f}")
print(f"Pose detection: every {process_every_n_frames} frames")
print("Press 'q' to close the video.")

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
pose_checks = 0
last_landmarks = None

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

            pose_checks += 1

            if result.pose_landmarks:
                last_landmarks = result.pose_landmarks[0]
            else:
                last_landmarks = None

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

            status = "POSE DETECTED"
            status_color = (0, 255, 0)
        else:
            status = "NO POSE DETECTED"
            status_color = (0, 0, 255)

        cv2.putText(
            resized_frame,
            status,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
        )

        cv2.putText(
            resized_frame,
            f"Frame: {frame_number} | AI checks: {pose_checks}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            resized_frame,
            "Optimized: pose every 3 frames | Q: Quit",
            (15, 85),
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