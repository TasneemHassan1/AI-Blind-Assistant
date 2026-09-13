"""Web dashboard for the AI Blind Assistant.

This is a MONITORING view only, for a sighted person (you, a demo audience,
a supervisor) to watch what the assistant is doing. It changes nothing
about how the assistant works: the user still activates it by saying the
wake word, asks their question out loud, and hears the answer spoken back
via voice.py — exactly like app.py. There is no text box anywhere, on
purpose, because a blind user cannot use one.

Run with:
    python web.py
Then open:
    http://127.0.0.1:5003
"""

from __future__ import annotations

import time
from threading import Thread

import cv2
from flask import Flask, Response, jsonify, render_template

from camera import Camera
from controller import AssistantController
from voice import speak
from voice_commands import VoiceCommandListener

app = Flask(__name__)

controller = AssistantController(speak=speak)
camera = Camera(controller, show_window=False)
commands = VoiceCommandListener(
    on_wake=controller.begin_question,
    on_question=controller.handle_question,
    on_cancel=controller.cancel_question,
)


def _generate_mjpeg():
    """Yield the latest annotated frame as a multipart JPEG stream."""
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify(controller.get_dashboard_state())


def main() -> None:
    commands.start()
    camera_thread = Thread(target=camera.run, daemon=True, name="camera")
    camera_thread.start()
    try:
        # use_reloader=False is required: the Flask reloader spawns a
        # second process, which would open the camera and microphone twice.
        app.run(host="0.0.0.0", port=5003, debug=False, threaded=True, use_reloader=False)
    finally:
        commands.stop()
        controller.shutdown()


if __name__ == "__main__":
    main()