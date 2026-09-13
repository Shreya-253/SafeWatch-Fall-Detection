import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

image_path = "outputs/screenshots/pose_test.jpg"
model_path = "models/pose_landmarker_lite.task"
window_name = "SafeWatch - Day 6 Pose Detection"

image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not open image: {image_path}")
    print("Check that you renamed a screenshot to pose_test.jpg")
    raise SystemExit

base_options = python.BaseOptions(model_asset_path=model_path)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)

with vision.PoseLandmarker.create_from_options(options) as landmarker:
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    )

    result = landmarker.detect(mp_image)

if result.pose_landmarks:
    landmarks = result.pose_landmarks[0]

    for index, landmark in enumerate(landmarks):
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])

        cv2.circle(image, (x, y), 8, (0, 255, 0), -1)

        if index in [11, 12, 23, 24, 25, 26, 27, 28]:
            cv2.putText(
                image,
                str(index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,    
            )

    print(f"Pose detected successfully: {len(landmarks)} landmarks found.")
else:
    print("No pose detected in this image.")

frame_height, frame_width = image.shape[:2]

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

if frame_height > frame_width:
    cv2.resizeWindow(window_name, 540, 960)
else:
    cv2.resizeWindow(window_name, 1280, 720)

cv2.imshow(window_name, image)

print("Press any key in the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()