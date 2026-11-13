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

### Notes

- The datasets do not link members to customers, so `app/members.json` assigns each member a
  `customer_id` from `product_model/data/merged_dataset.csv`. Edit it as needed.
- Say one of the training phrases ("Yes, approve" / "Confirm transaction") for the voice check.
- The voice model is a closed-set classifier over the four members: it always picks one of them,
  so it detects a mismatch with the claimed name but cannot reject an unknown voice.
- The models were saved with scikit-learn 1.6.1 (voice), 1.9.0 (face) and 1.9.1 (product); the CLI
  hides the resulting version warnings.
