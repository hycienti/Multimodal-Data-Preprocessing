"""Grab a photo from the webcam and a recording from the microphone."""
import time

import numpy as np

from voice_recognition.audio_features import SAMPLE_RATE


class CaptureError(RuntimeError):
    pass


def capture_photo(camera_index=0, preview=True, warmup_frames=15, note=None):
    """Return one BGR frame from the webcam.

    preview=True opens a window: SPACE captures, Q / ESC cancels.
    preview=False lets the camera settle for a few frames and grabs one.
    note is an extra line drawn on the preview (e.g. "Attempt 2 of 3").
    """
    import cv2

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise CaptureError(
            f"Could not open camera {camera_index}. On macOS, allow camera access for your "
            "terminal in System Settings > Privacy & Security > Camera."
        )
    window = "Face verification - SPACE to capture, Q to cancel"
    try:
        if not preview:
            frame = None
            for _ in range(warmup_frames):
                ok, frame = camera.read()
            if frame is None or not ok:
                raise CaptureError("Camera returned no frame")
            return frame

        while True:
            ok, frame = camera.read()
            if not ok:
                raise CaptureError("Camera returned no frame")
            shown = frame.copy()
            cv2.putText(shown, "SPACE: capture   Q: cancel", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if note:
                cv2.putText(shown, note, (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                return frame
            if key in (ord("q"), 27):
                raise CaptureError("Photo capture cancelled")
    finally:
        camera.release()
        if preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # lets macOS actually close the window


def record_audio(seconds=4.0, countdown=3, device=None):
    """Record mono float32 audio at the voice model's sample rate."""
    import sounddevice as sd

    for remaining in range(countdown, 0, -1):
        print(f"  Recording in {remaining}...", flush=True)
        time.sleep(1)
    print(f"  Recording for {seconds:g}s - speak now", flush=True)
    try:
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=device)
        sd.wait()
    except Exception as error:  # PortAudio raises a variety of errors
        raise CaptureError(
            f"Could not record from microphone: {error}. On macOS, allow microphone access for "
            "your terminal in System Settings > Privacy & Security > Microphone."
        ) from error
    print("  Done.", flush=True)

    audio = audio.flatten()
    if np.max(np.abs(audio)) < 1e-4:
        raise CaptureError("Recording is silent. Check the microphone input.")
    return audio
