# Multimodal-Data-Preprocessing
Multimodal Data Preprocessing 

## Authentication CLI

`main.py` runs the full flow from the webcam and microphone:

```
Start -> Facial recognition -> Product recommendation -> Voice validation -> Predicted product
              | fail                                          | fail
         ACCESS DENIED                                   ACCESS DENIED
```

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install facenet-pytorch==2.6.0 --no-deps
```

The first run downloads the FaceNet (VGGFace2) weights. On macOS, give your terminal camera and
microphone access in System Settings > Privacy & Security.

### Run

```bash
python main.py                       # prompts for username, opens camera (SPACE to capture), then records 4 s
python main.py --name bode           # skip the prompt
python main.py --name bode --image photo.jpg --audio voice.wav   # from files, no hardware
```

| Option | Meaning |
|---|---|
| `--camera N` | camera index (default 0) |
| `--no-preview` | grab a frame without opening a preview window |
| `--seconds S` | recording length (default 4) |
| `--attempts N` | tries per check before access is denied (default 3) |
| `--customer-id ID` | use this customer's profile instead of the mapping |
| `--save-captures` | keep the photo and recording in `captures/` |

Exit code is 0 when access is granted and 1 when it is denied.

### What you'll see

The CLI explains each step as it goes, so a first-time user can follow it without this README:

```
Hi hycient. This login has 3 steps:
  1. Face check   - we compare a photo of you with your account
  2. Product pick - we prepare your recommendation (hidden for now)
  3. Voice check  - you say a short phrase, then we show the result

=== Step 1/3 - Facial recognition ===
  A camera window is opening so we can take your photo.
  Look at the camera and press SPACE to take the photo (Q or ESC cancels).
  Photo taken. Checking your face... (this can take a few seconds)
  Face verified as hycient (similarity 0.814)
  Next: preparing your product recommendation.

=== Step 2/3 - Product recommendation ===
  Your recommendation is ready (customer 188).
  It stays hidden until your voice confirms it's you.
  Next: the voice check.

=== Step 3/3 - Voice validation ===
  No camera this time - we only record your voice for 4 seconds.
  After the countdown, say "Yes, approve" or "Confirm transaction".
  Recording in 3...
  ...
  Checking your voice...
  Voice verified as hycient

=== Predicted product ===
  Clothing  (confidence 50%)
  runner-up: Sports (20%)

  ACCESS GRANTED - your face and voice both matched.
```

When a check fails, the CLI says why, how many tries are left and that it is opening the camera
(or recording) again. The camera window also shows the attempt number:

```
  [1/3] Face does not match this account.
  2 tries left. Opening the camera again - face it directly, in good light.
```

After the last failed try, or if you cancel with Q or ESC, the run ends with `ACCESS DENIED` and
the reason. Nothing about the product is shown.

### Notes

- The datasets do not link members to customers, so `app/members.json` assigns each member a
  `customer_id` from `product_model/data/merged_dataset.csv`. Edit it as needed.
- Say one of the training phrases ("Yes, approve" / "Confirm transaction") for the voice check.
- The voice model is a closed-set classifier over the four members: it always picks one of them,
  so it detects a mismatch with the claimed name but cannot reject an unknown voice.
- The models were saved with scikit-learn 1.6.1 (voice), 1.9.0 (face) and 1.9.1 (product); the CLI
  hides the resulting version warnings.
