import cv2

video_path = "data/test_videos/test_video.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print(f"Error: Could not open video: {video_path}")
    raise SystemExit

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if fps == 0:
    print("Error: FPS could not be read from the video.")
    video.release()
    raise SystemExit

duration_seconds = total_frames / fps

window_name = "SafeWatch - Day 4 Video Information"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if frame_height > frame_width:
    cv2.resizeWindow(window_name, 540, 960)
else:
    cv2.resizeWindow(window_name, 1280, 720)

print(f"Resolution: {frame_width} x {frame_height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration_seconds:.2f} seconds")
print("Press 'q' to close the video.")

while True:
    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    current_time = current_frame / fps
    progress = (current_frame / total_frames) * 100

    cv2.putText(
        frame,
        "SafeWatch - Day 4: Video Information",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Frame: {current_frame}/{total_frames}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Time: {current_time:.1f}s / {duration_seconds:.1f}s",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Resolution: {frame_width}x{frame_height} | FPS: {fps:.2f}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    progress_bar_width = 400
    progress_bar_height = 18
    progress_bar_x = 20
    progress_bar_y = 160

    cv2.rectangle(
        frame,
        (progress_bar_x, progress_bar_y),
        (progress_bar_x + progress_bar_width, progress_bar_y + progress_bar_height),
        (255, 255, 255),
        2,
    )

    filled_width = int((progress / 100) * progress_bar_width)

    cv2.rectangle(
        frame,
        (progress_bar_x, progress_bar_y),
        (progress_bar_x + filled_width, progress_bar_y + progress_bar_height),
        (0, 255, 0),
        -1,
    )

    cv2.putText(
        frame,
        f"Progress: {progress:.1f}%",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        print("Video closed by user.")
        break

video.release()
cv2.destroyAllWindows()