from flask import Flask, Response
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)

def generate():
    while True:
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes() +
            b'\r\n'
        )

@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

app.run(host="0.0.0.0", port=5000)
