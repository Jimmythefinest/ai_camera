import numpy as np
import cv2
import requests
from ultralytics import YOLO

# ---- Config: match your ESP32 sketch ----
ESP32_URL = "http://10.16.111.224/stream"  # <-- change to your ESP32's IP
FRAME_W, FRAME_H = 160, 120
FRAME_BYTES = FRAME_W * FRAME_H * 2  # RGB565 = 2 bytes/pixel
DISPLAY_SCALE = 4  # upscale for viewing since native frame is tiny

HEADER_END = b"\r\n\r\n"

model = YOLO("yolo11n.pt")
# model.to("cuda")


def rgb565_to_bgr(payload: bytes) -> np.ndarray:
    """Convert one raw RGB565 (big-endian) frame buffer into a BGR image for OpenCV."""
    arr = np.frombuffer(payload, dtype=">u2", count=FRAME_W * FRAME_H).reshape(FRAME_H, FRAME_W)
    r = ((arr >> 11) & 0x1F).astype(np.uint16)
    g = ((arr >> 5) & 0x3F).astype(np.uint16)
    b = (arr & 0x1F).astype(np.uint16)
    r = (r * 255 // 31).astype(np.uint8)
    g = (g * 255 // 63).astype(np.uint8)
    b = (b * 255 // 31).astype(np.uint8)
    return np.dstack((b, g, r))  # OpenCV expects BGR order


def frame_generator(url: str):
    """Connects once and yields raw RGB565 payloads as they arrive in the multipart stream."""
    resp = requests.get(url, stream=True, timeout=10)
    resp.raise_for_status()
    buf = b""
    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue
        buf += chunk
        while True:
            header_end = buf.find(HEADER_END)
            if header_end == -1:
                break
            payload_start = header_end + 4
            if len(buf) < payload_start + FRAME_BYTES:
                break
            payload = buf[payload_start:payload_start + FRAME_BYTES]
            buf = buf[payload_start + FRAME_BYTES:]
            yield payload


def main():
    print(f"Connecting to {ESP32_URL} ...")
    try:
        gen = frame_generator(ESP32_URL)
    except requests.exceptions.RequestException as e:
        print(f"Could not connect to ESP32 stream: {e}")
        return

    for payload in gen:
        frame = rgb565_to_bgr(payload)

        # YOLO on the native small frame; upscale afterwards for display/annotation clarity
        results = model(frame, conf=0.5, verbose=False)
        annotated = results[0].plot()

        annotated = cv2.resize(
            annotated,
            (FRAME_W * DISPLAY_SCALE, FRAME_H * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("YOLO ESP32-CAM", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()