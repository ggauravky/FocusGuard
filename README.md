# FocusGuard (OpenCV + Python)

FocusGuard is a real-time webcam monitor that detects:

- Face
- Eyes
- Smile
- Lips

And adds a behavior rule:

- If the user appears focused on the screen continuously for 3-4 seconds, trigger a warning/alarm.

## How It Works

The app combines feature detection with a time-based attention heuristic.

Detection layer:

- Face, eyes, smile: OpenCV Haar cascades.
- Lips: lower-face HSV color segmentation.

Attention layer:

- Pick the primary face (largest face box).
- Confirm at least one eye is detected inside that face.
- Check if the face center is in a central window of the frame.
- If all true continuously for threshold seconds (default 3.5s), fire warning/alarm.

This is a practical heuristic, not a medical-grade gaze estimator.

## Project Structure

```text
FocusGuard/
  README.md
  requirements.txt
  .gitignore
  src/
    app.py
    attention/
      ATTENTION_LOGIC.md
      __init__.py
      screen_attention_monitor.py
    detectors/
      __init__.py
      haar_feature_detector.py
    utils/
      __init__.py
      drawing.py
```

## Setup

1. Install dependencies with system Python:

```powershell
py -3.13 -m pip install -r requirements.txt
```

## Run

Basic:

```powershell
py -3.13 -m src.app
```

Custom camera and threshold:

```powershell
py -3.13 -m src.app --camera 0 --width 1280 --height 720 --attention-threshold 3.5 --center-ratio 0.62
```

## Runtime Display

- Bounding boxes for face, eyes, smile, and lips.
- Focus timer overlay, for example: `Focus timer: 2.8s / 3.5s`.
- State text, for example: `Focused`, `No face`, `Eyes not visible`, `Face off-center`.
- Alarm when threshold is crossed.

## Tunable Parameters

- `--attention-threshold`: seconds before warning, default `3.5`.
- `--center-ratio`: central valid area size, default `0.62`.
  - Smaller value = stricter center constraint.
  - Larger value = more tolerant head position.

## Platform Note

- On Windows, alarm uses `winsound.Beep`.
- If beep API is unavailable, the app falls back to terminal bell/message.

## Controls

- Press `q` to quit.

## Practical Tips

- Use good front lighting for stable face/eye detection.
- Keep camera at eye level.
- Increase `--attention-threshold` if alerts fire too quickly.
- Increase `--center-ratio` if alerts miss when you are slightly off-center.

## Internal Docs

- Attention logic details: `src/attention/ATTENTION_LOGIC.md`
