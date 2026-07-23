"""
Security camera fleet server.

- Each registered ESP32-CAM gets its own CameraWorker thread that connects to
  its raw RGB565 multipart stream, decodes frames, and does cheap motion
  detection.
- Motion-triggered frames are handed off to a single shared InferenceWorker
  thread running one YOLO model instance, so CPU-limited hardware (e.g. a
  Pi 4) isn't running detection on every frame from every camera.
- Flask serves a dashboard to register/remove cameras, an MJPEG feed per
  camera for browser viewing, and a single-JPEG /snapshot endpoint for a
  backend that wants to poll on its own schedule instead of consuming a
  live stream.
- When a person is detected, a WhatsApp alert (image + caption) is sent via
  the REST API from api-server.js, gated by a per-camera cooldown so it
  doesn't spam on every frame while someone is in view.
"""

import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
FRAME_W, FRAME_H = 160, 120          # must match the ESP32 sketch's FRAME_W/FRAME_H
FRAME_BYTES = FRAME_W * FRAME_H * 2  # RGB565 = 2 bytes/pixel
HEADER_END = b"\r\n\r\n"

MOTION_PIXEL_THRESHOLD = 500   # changed pixels needed to call it "motion"
MOTION_DIFF_THRESHOLD = 25     # per-pixel grayscale delta to count as "changed"
FRAME_SKIP = 3                 # only submit every Nth motion frame to YOLO
INFERENCE_IMGSZ = 320          # smaller = faster on CPU-only hardware
CONF_THRESHOLD = 0.5

CAMERAS_FILE = os.path.join(os.path.dirname(__file__), "cameras.json")
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolo11n.pt")

# --- WhatsApp alerting (talks to the api-server.js REST API) --------------
ALERT_CLASSES = {"person"}  # which detected classes should trigger an alert
ALERT_COOLDOWN_SECONDS = float(os.environ.get("ALERT_COOLDOWN_SECONDS", 60))
WHATSAPP_API_URL = os.environ.get("WHATSAPP_API_URL", "http://localhost:3000")
WHATSAPP_TO = os.environ.get("WHATSAPP_TO", "263789878698")          # phone number, digits only
WHATSAPP_API_KEY = os.environ.get("WHATSAPP_API_KEY", "")  # optional, matches API_KEY in api-server.js
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", os.path.join(os.path.dirname(__file__), "alert_snapshots"))
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

app = Flask(__name__)

# --------------------------------------------------------------------------
# Shared model + inference queue
# --------------------------------------------------------------------------
model = YOLO(MODEL_PATH)
inference_queue = queue.Queue(maxsize=16)


# --------------------------------------------------------------------------
# RGB565 -> BGR (same math as the JS canvas decoder / earlier script)
# --------------------------------------------------------------------------
def rgb565_to_bgr(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=">u2", count=FRAME_W * FRAME_H).reshape(FRAME_H, FRAME_W)
    r = ((arr >> 11) & 0x1F).astype(np.uint16)
    g = ((arr >> 5) & 0x3F).astype(np.uint16)
    b = (arr & 0x1F).astype(np.uint16)
    r = (r * 255 // 31).astype(np.uint8)
    g = (g * 255 // 63).astype(np.uint8)
    b = (b * 255 // 31).astype(np.uint8)
    return np.dstack((b, g, r))


# --------------------------------------------------------------------------
# WhatsApp alert helper
# --------------------------------------------------------------------------
def send_whatsapp_alert(cam_id: str, jpeg_bytes: bytes, classes: list):
    """Fire-and-forget: save the frame to disk and POST it to the WhatsApp
    REST API. Runs in its own thread so a slow/broken network call never
    stalls the inference loop."""

    if not WHATSAPP_TO:
        print("[alert] WHATSAPP_TO is not set, skipping alert")
        return

    def _worker():
        timestamp = int(time.time())
        filename = f"{cam_id}_{timestamp}.jpg"
        filepath = os.path.join(SNAPSHOT_DIR, filename)
        try:
            with open(filepath, "wb") as f:
                f.write(jpeg_bytes)
        except OSError as e:
            print(f"[alert] failed to write snapshot for {cam_id}: {e}")
            return

        caption = f"Person detected on camera '{cam_id}' ({', '.join(classes)})"
        headers = {"Content-Type": "application/json"}
        if WHATSAPP_API_KEY:
            headers["X-API-Key"] = WHATSAPP_API_KEY

        try:
            resp = requests.post(
                f"{WHATSAPP_API_URL}/api/send-image",
                json={"to": WHATSAPP_TO, "path": filepath, "caption": caption},
                headers=headers,
                timeout=15,
            )
            if resp.ok:
                print(f"[alert] sent WhatsApp alert for {cam_id}")
            else:
                print(f"[alert] WhatsApp API returned {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[alert] failed to reach WhatsApp API for {cam_id}: {e}")

    threading.Thread(target=_worker, daemon=True).start()


# --------------------------------------------------------------------------
# Camera registry (persisted to disk)
# --------------------------------------------------------------------------
class CameraManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.cameras = {}  # cam_id -> dict of metadata + worker + latest jpeg
        self._load()

    def _load(self):
        if os.path.exists(CAMERAS_FILE):
            with open(CAMERAS_FILE) as f:
                saved = json.load(f)
            for cam_id, info in saved.items():
                self.add(cam_id, info["ip"], persist=False, start=True)

    def _persist(self):
        with self.lock:
            data = {cid: {"ip": c["ip"]} for cid, c in self.cameras.items()}
        with open(CAMERAS_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def add(self, cam_id, ip, persist=True, start=True):
        with self.lock:
            if cam_id in self.cameras:
                return False
            self.cameras[cam_id] = {
                "ip": ip,
                "status": "connecting",
                "last_frame_time": None,
                "last_detection": None,
                "last_detection_time": None,
                "latest_jpeg": None,
                "worker": None,
                "last_alert_time": 0.0,
            }
        if start:
            worker = CameraWorker(cam_id, ip, self)
            with self.lock:
                self.cameras[cam_id]["worker"] = worker
            worker.start()
        if persist:
            self._persist()
        return True

    def remove(self, cam_id):
        with self.lock:
            cam = self.cameras.pop(cam_id, None)
        if cam and cam["worker"]:
            cam["worker"].stop()
        self._persist()
        return cam is not None

    def update(self, cam_id, **fields):
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id].update(fields)

    def get(self, cam_id):
        with self.lock:
            return dict(self.cameras.get(cam_id, {}))

    def try_start_alert(self, cam_id, cooldown_seconds):
        """Atomically checks whether enough time has passed since the last
        alert for this camera, and if so, marks 'now' as the last alert time
        and returns True. Returns False if still in cooldown."""
        now = time.time()
        with self.lock:
            cam = self.cameras.get(cam_id)
            if cam is None:
                return False
            if now - cam.get("last_alert_time", 0.0) < cooldown_seconds:
                return False
            cam["last_alert_time"] = now
            return True

    def list_public(self):
        with self.lock:
            out = {}
            for cid, c in self.cameras.items():
                out[cid] = {
                    "ip": c["ip"],
                    "status": c["status"],
                    "last_frame_time": c["last_frame_time"],
                    "last_detection": c["last_detection"],
                    "last_detection_time": c["last_detection_time"],
                    "last_alert_time": c.get("last_alert_time") or None,
                }
            return out


manager = CameraManager()


# --------------------------------------------------------------------------
# Per-camera worker: connects to the raw stream, decodes frames, motion-gates
# --------------------------------------------------------------------------
class CameraWorker(threading.Thread):
    def __init__(self, cam_id, ip, manager: CameraManager):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.ip = ip
        self.manager = manager
        self._stop_flag = threading.Event()
        self.prev_gray = None
        self.frame_counter = 0

    def stop(self):
        self._stop_flag.set()

    def run(self):
        url = f"http://{self.ip}/stream"
        while not self._stop_flag.is_set():
            try:
                self.manager.update(self.cam_id, status="connecting")
                resp = requests.get(url, stream=True, timeout=10)
                resp.raise_for_status()
                self.manager.update(self.cam_id, status="connected")
                buf = b""
                for chunk in resp.iter_content(chunk_size=4096):
                    if self._stop_flag.is_set():
                        return
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
                        self._handle_frame(payload)
            except requests.exceptions.RequestException:
                self.manager.update(self.cam_id, status="disconnected")
                time.sleep(3)  # backoff before reconnect attempt

    def _handle_frame(self, payload: bytes):
        frame = rgb565_to_bgr(payload)

        # Always refresh the "live" jpeg so the dashboard looks live even
        # when nothing is currently triggering detection.
        ok, jpeg = cv2.imencode(".jpg", frame)
        if ok:
            self.manager.update(
                self.cam_id,
                latest_jpeg=jpeg.tobytes(),
                last_frame_time=time.time(),
            )

        # Cheap motion gate before ever touching YOLO.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = False
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            changed = int(np.count_nonzero(diff > MOTION_DIFF_THRESHOLD))
            motion = changed > MOTION_PIXEL_THRESHOLD
        self.prev_gray = gray

        if not motion:
            return

        self.frame_counter += 1
        if self.frame_counter % FRAME_SKIP != 0:
            return

        try:
            inference_queue.put_nowait((self.cam_id, frame))
        except queue.Full:
            pass  # inference thread is behind; drop this frame rather than backlog


# --------------------------------------------------------------------------
# Shared inference thread
# --------------------------------------------------------------------------
class InferenceWorker(threading.Thread):
    def __init__(self, manager: CameraManager):
        super().__init__(daemon=True)
        self.manager = manager

    def run(self):
        while True:
            cam_id, frame = inference_queue.get()
            results = model(frame, conf=CONF_THRESHOLD, imgsz=INFERENCE_IMGSZ, verbose=False)
            annotated = results[0].plot()
            ok, jpeg = cv2.imencode(".jpg", annotated)
            if not ok:
                continue

            classes = []
            names = results[0].names
            for box in results[0].boxes:
                classes.append(names[int(box.cls[0])])

            self.manager.update(
                cam_id,
                latest_jpeg=jpeg.tobytes(),
                last_detection=classes if classes else None,
                last_detection_time=time.time() if classes else self.manager.get(cam_id).get("last_detection_time"),
            )

            if ALERT_CLASSES.intersection(classes):
                if self.manager.try_start_alert(cam_id, ALERT_COOLDOWN_SECONDS):
                    send_whatsapp_alert(cam_id, jpeg.tobytes(), classes)


# --------------------------------------------------------------------------
# Flask routes
# --------------------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("Dashboard.html")


@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    return jsonify(manager.list_public())


@app.route("/api/cameras", methods=["POST"])
def add_camera():
    data = request.get_json(force=True)
    cam_id = data.get("id", "").strip()
    ip = data.get("ip", "").strip()
    if not cam_id or not ip:
        return jsonify({"error": "id and ip are required"}), 400
    if not manager.add(cam_id, ip):
        return jsonify({"error": "camera id already exists"}), 409
    return jsonify({"ok": True})


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def delete_camera(cam_id):
    if not manager.remove(cam_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


def _mjpeg_generator(cam_id):
    while True:
        cam = manager.get(cam_id)
        jpeg = cam.get("latest_jpeg")
        if jpeg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        time.sleep(0.1)  # ~10 fps ceiling for the browser feed regardless of source fps


@app.route("/feed/<cam_id>")
def feed(cam_id):
    if not manager.get(cam_id):
        return "camera not found", 404
    return Response(_mjpeg_generator(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot/<cam_id>")
def snapshot(cam_id):
    cam = manager.get(cam_id)
    if not cam:
        return "camera not found", 404
    jpeg = cam.get("latest_jpeg")
    if not jpeg:
        return "no frame yet", 503
    return Response(jpeg, mimetype="image/jpeg")


# --------------------------------------------------------------------------
# Boot
# --------------------------------------------------------------------------
if __name__ == "__main__":
    InferenceWorker(manager).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)