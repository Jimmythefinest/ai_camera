import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")
# model.to("cuda")
# Open the built-in webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, conf=0.5)

    # Draw detections
    annotated = results[0].plot()

    cv2.imshow("YOLO Webcam", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
