"""Multimodal authentication CLI.

Start -> facial recognition -> product recommendation -> voice validation -> show product.
A failed face or voice check ends with ACCESS DENIED.

    python main.py                       # live camera + microphone
    python main.py --name bode --image face.jpg --audio voice.wav   # from files
"""
import argparse
import asyncio
import sys
import warnings
from datetime import datetime
from pathlib import Path

# The three models were saved with different scikit-learn patch versions (1.6.1, 1.9.0, 1.9.1);
# they load and predict correctly, so keep the CLI output readable.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
VOICE_MODEL = ROOT / "voice_recognition" / "models" / "voice_verification_model.joblib"
CAPTURES_DIR = ROOT / "captures"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Face + voice gated product recommendation")
    parser.add_argument("--name", help="who is logging in (prompted if omitted)")
    parser.add_argument("--image", help="use this image instead of the camera")
    parser.add_argument("--audio", help="use this audio file instead of the microphone")
    parser.add_argument("--customer-id", type=int, help="override the member -> customer mapping")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--no-preview", action="store_true", help="grab a frame without a preview window")
    parser.add_argument("--seconds", type=float, default=4.0, help="recording length (default 4)")
    parser.add_argument("--attempts", type=int, default=3, help="tries per check (default 3)")
    parser.add_argument("--save-captures", action="store_true", help=f"save photo/audio to {CAPTURES_DIR.name}/")
    args = parser.parse_args(argv)
    args.attempts = max(1, args.attempts)
    return args


def banner(text):
    print(f"\n=== {text} ===")


def access_denied(reason):
    print(f"\n  ACCESS DENIED: {reason}")
    print("  Nothing is shown without both checks. Run the command again to try once more.")
    return 1


def show_overview(name):
    print(f"\nHi {name}. This login has 3 steps:")
    print("  1. Face check   - we compare a photo of you with your account")
    print("  2. Product pick - we prepare your recommendation (hidden for now)")
    print("  3. Voice check  - you say a short phrase, then we show the result")


def tries_left(attempt, attempts):
    left = attempts - attempt
    return f"{left} {'try' if left == 1 else 'tries'} left"


def save_capture(kind, data):
    CAPTURES_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if kind == "photo":
        import cv2
        path = CAPTURES_DIR / f"face-{stamp}.jpg"
        cv2.imwrite(str(path), data)
    else:
        import soundfile as sf
        from voice_recognition.audio_features import SAMPLE_RATE
        path = CAPTURES_DIR / f"voice-{stamp}.wav"
        sf.write(path, data, SAMPLE_RATE)
    print(f"  saved {path.relative_to(ROOT)}")


def run_face_check(args, name):
    from app.capture import capture_photo
    from facial_recognition.verify_user import verify_user

    banner("Step 1/3 - Facial recognition")
    for attempt in range(1, args.attempts + 1):
        if args.image:
            print(f"  Using the photo {args.image} instead of the camera.")
            image, color_order = args.image, "RGB"
        else:
            if args.no_preview:
                print("  Look at the camera. The photo is taken automatically.")
            else:
                if attempt == 1:
                    print("  A camera window is opening so we can take your photo.")
                print("  Look at the camera and press SPACE to take the photo (Q or ESC cancels).")
            image = capture_photo(args.camera, preview=not args.no_preview,
                                  note=f"Attempt {attempt} of {args.attempts}")
            color_order = "BGR"
            if args.save_captures:
                save_capture("photo", image)

        taken = "" if args.image else "Photo taken. "
        slow = " (this can take a few seconds)" if attempt == 1 else ""
        print(f"  {taken}Checking your face...{slow}")
        result = asyncio.run(verify_user(user_name=name, image=image, color_order=color_order))
        if result["success"]:
            match = result["matches"]
            print(f"  Face verified as {name} (similarity {match['similarity_score']})")
            print("  Next: preparing your product recommendation.")
            return True, None
        # the pipeline's messages end in "Try again", which is wrong on the last attempt
        message = result["message"].removesuffix(" Try again")
        print(f"  [{attempt}/{args.attempts}] {message}")
        if args.image:  # same file would give the same answer
            break
        if attempt < args.attempts:
            print(f"  {tries_left(attempt, args.attempts)}. Opening the camera again - "
                  "face it directly, in good light.")
    return False, message


def run_product_model(args, name):
    from app.customers import customer_id_for, customer_profile
    from product_model.product_model import predict_with_confidence

    banner("Step 2/3 - Product recommendation")
    customer_id = args.customer_id or customer_id_for(name)
    prediction = predict_with_confidence(customer_profile(customer_id))
    print(f"  Your recommendation is ready (customer {customer_id}).")
    print("  It stays hidden until your voice confirms it's you.")
    print("  Next: the voice check.")
    return prediction


def run_voice_check(args, name):
    from app.capture import CaptureError, record_audio
    from voice_recognition.audio_features import extract_features, load_audio
    from voice_recognition.models.voice_verification import VoiceVerification

    banner("Step 3/3 - Voice validation")
    verifier = VoiceVerification(VOICE_MODEL)
    for attempt in range(1, args.attempts + 1):
        if args.audio:
            print(f"  Using the recording {args.audio} instead of the microphone.")
            try:
                signal = load_audio(args.audio)
            except Exception as error:  # librosa/soundfile raise their own error types
                raise CaptureError(f"Could not read the audio file {args.audio}: {error}") from error
        else:
            if attempt == 1:
                print(f"  No camera this time - we only record your voice for {args.seconds:g} seconds.")
            print('  After the countdown, say "Yes, approve" or "Confirm transaction".')
            signal = record_audio(args.seconds)
            if args.save_captures:
                save_capture("voice", signal)

        print("  Checking your voice...")
        try:
            result = verifier.verify(extract_features(signal), name)
        except ValueError as error:
            message = str(error)
        else:
            if result["verified"]:
                print(f"  Voice verified as {name}")
                return True, None
            message = f"voice recognised as '{result['predicted_identity']}', not '{name}'"
        print(f"  [{attempt}/{args.attempts}] {message}")
        if args.audio:
            break
        if attempt < args.attempts:
            print(f"  {tries_left(attempt, args.attempts)}. Recording again - "
                  "speak clearly, close to the microphone.")
    return False, message


def show_product(prediction):
    banner("Predicted product")
    print(f"  {prediction['product']}  (confidence {prediction['confidence']:.0%})")
    print(f"  runner-up: {prediction['runner_up']} ({prediction['runner_up_confidence']:.0%})")
    if prediction["low_confidence"]:
        print("  note: low confidence, treat as a suggestion")


def main(argv=None):
    args = parse_args(argv)
    name = (args.name or input("Username: ")).strip().lower()
    if not name:
        return access_denied("no username given")

    for label, path in (("Photo", args.image), ("Audio", args.audio)):
        if path and not Path(path).is_file():
            return access_denied(f"{label} file not found: {path}")

    show_overview(name)

    from app.capture import CaptureError
    try:
        ok, reason = run_face_check(args, name)
        if not ok:
            return access_denied(reason)

        prediction = run_product_model(args, name)

        ok, reason = run_voice_check(args, name)
        if not ok:
            return access_denied(reason)
    except (CaptureError, KeyError, FileNotFoundError) as error:
        return access_denied(str(error).strip("'\""))
    except KeyboardInterrupt:
        return access_denied("cancelled")

    show_product(prediction)
    print("\n  ACCESS GRANTED - your face and voice both matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
