import cv2
import os

video_path = "video1.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    filename = os.path.join(output_dir, f"frame_{frame_number:06d}.jpg")
    cv2.imwrite(filename, frame)

    print(f"Saved {filename}")

    frame_number += 1

cap.release()

print(f"Extracted {frame_number} frames.")
