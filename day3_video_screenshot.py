import os
from datetime import datetime

import cv2

video_path = "data/test_videos/test_video.mp4"
screenshot_folder = "outputs/screenshots"

os.makedirs(screenshot_folder, exist_ok=True)

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"Error: Could not open the video file: {video_path}")
    print("Check that the file exists and is named test_video.mp4")
    raise SystemExit
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print(f"Original video size: {frame_width} x {frame_height}")
print(f"Video FPS: {fps}")

window_name = "SafeWatch - Video Player"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if frame_height > frame_width:
    cv2.resizeWindow(window_name, 540, 960)
else:
    cv2.resizeWindow(window_name, 1280, 720)
print("Video started.")
print("Press 's' to save a screenshot.")
print("Press 'q' to close the video.")

while True:
    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    cv2.putText(
        frame,
        "SafeWatch - Day 3 Video Test",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        "Press S: Save Screenshot | Press Q: Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = os.path.join(
            screenshot_folder,
            f"screenshot_{timestamp}.jpg",
        )

        cv2.imwrite(screenshot_path, frame)
        print(f"Screenshot saved: {screenshot_path}")

    elif key == ord("q"):
        print("Video closed by user.")
        break

video.release()
cv2.destroyAllWindows()