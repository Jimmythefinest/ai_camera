import cv2
import os
import glob

from rtmlib import Wholebody, draw_skeleton

# Folder containing extracted frames
frames_dir = "frames"

# Get all frame filenames in order
frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))

# Load RTMPose
wholebody = Wholebody(
    mode="performance",
    backend="onnxruntime",
    device="cuda"
)

for frame_file in frame_files:

    image = cv2.imread(frame_file)

    if image is None:
        continue

    # Detect pose
    keypoints, scores = wholebody(image)

    # Draw skeleton
    output = draw_skeleton(image.copy(), keypoints, scores)

    # Show filename
    cv2.putText(
        output,
        os.path.basename(frame_file),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("RTMPose", output)

    # Wait 30 ms (about 33 FPS)
    key = cv2.waitKey(1)

    # ESC quits
    if key == 27:
        break

cv2.destroyAllWindows()
