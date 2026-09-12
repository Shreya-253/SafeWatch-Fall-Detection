import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open the webcam.")
    print("Close Zoom, Google Meet, Teams, Discord, or any app using the camera.")
    raise SystemExit

print("Webcam started. Press 'q' in the video window to close.")

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read a frame from the webcam.")
        break

    cv2.putText(
        frame,
        "SafeWatch - Webcam is Working!",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("SafeWatch - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Webcam closed safely.")