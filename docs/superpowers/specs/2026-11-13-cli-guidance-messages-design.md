# CLI guidance messages - design

## Intent

Someone running `main.py` should always know what they are looking at, what the program wants
from them, and what happens next. Today the run shows bare step headers and one-line results, so
people don't know why the camera opened (or reopened), whether the program is frozen, or what to
do after a denial.

Audience: a recorded demo video, and a few first-time users running it themselves.

## Goals

- Guidance in all three parts of the run: camera (face check), voice check, product recommendation.
- Short, plain lines that a first-time user can follow alone and that read well on a recording.
- Always on. No flag to enable it.

## Non-goals

- No change to how the face, voice or product checks work, their order, thresholds, attempts or
  exit codes.
- No new options, colours or dependencies.

## Anti-goals

- Walls of text. Each moment gets one or two lines.
- Internal wording ("locked until voice check", "closed-set", model names).

## Approaches considered

- **A. Guidance always on, in the terminal** - chosen. One behaviour to record; nothing for a new
  user to switch on. Cost: repeat users see a few extra lines.
- **B. Short by default, full guidance behind `--guide`** - rejected. First-time users are the ones
  who won't know the flag exists.
- **C. Guidance inside the camera window** - partly adopted. The window shows the attempt number,
  because that is where people look during capture. On its own it can't cover the voice step or
  the pauses.

## Design

All text changes live in `main.py` and `app/capture.py`.

1. **Overview at the start** (after the username is known), listing the three steps:
   face check, product pick (hidden for now), voice check then result.
2. **Camera (step 1)**
   - Before the window opens: say a camera window is opening, press SPACE to take the photo,
     Q or ESC cancels. With `--no-preview`: say the photo is taken automatically.
   - After capture: "Checking your face..." with a note that the first run can take a few seconds.
   - On a failed attempt: the reason, how many tries are left, and that the camera is opening
     again, with a tip (face the camera directly, good light).
   - The camera window's on-screen text shows "Attempt N of M".
   - On success: the existing verified line, then what comes next.
3. **Product recommendation (step 2)**: say the recommendation is ready and stays hidden until the
   voice check confirms identity.
4. **Voice (step 3)**
   - Before recording: no camera this time, only a voice recording of N seconds; after the
     countdown say one of the two phrases.
   - After recording: "Checking your voice...".
   - On a failed attempt: the reason, tries left, and that it will record again.
5. **Endings**
   - Granted: unchanged result block, plus one line saying both checks passed.
   - Denied: the reason, plus one line on what to do next (run the command again; for camera or
     microphone permission errors the existing hint stays).
6. **File mode** (`--image` / `--audio`): say which file is being checked instead of the
   camera/microphone lines. No retry text, since files are not retried.

## Left to the builder

- Exact wording, within the tone above.
- Whether the messages are plain `print` calls or a small helper.

## Checking it

No test suite exists. Verify by running: the granted path live, a wrong-name run using all
attempts, a cancelled capture, and the file-mode path with `--image`.
