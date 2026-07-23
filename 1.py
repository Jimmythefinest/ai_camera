from ultralytics import YOLO
import cv2

# Load a pose model
model = YOLO("yolo11n-pose.pt")

# Read image
image = cv2.imread("person1.jpeg")

# Run inference
results = model(image)

for result in results:

    # Each detected person
    for person in result.keypoints.data:

        print("Person")

        # Convert tensor to numpy
        points = person.cpu().numpy()

        for i, point in enumerate(points):
            x, y, confidence = point
            print(f"{i}: x={x:.1f}, y={y:.1f}, conf={confidence:.2f}")

            if confidence > 0.5:
                cv2.circle(image, (int(x), int(y)), 5, (0,255,0), -1)

cv2.imshow("Pose", image)
cv2.waitKey(0)
