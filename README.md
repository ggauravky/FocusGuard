# FocusGuard (OpenCV + Python)

FocusGuard is a real-time webcam monitor for this specific rule:

- Detect a phone in the camera view.
- Detect your face.
- If you appear to be looking at that phone continuously for more than 4 seconds, play a beep warning.

## How It Works

The app now uses a phone-first pipeline.

Detection layer:

- Face, eyes, smile, lips: OpenCV Haar/lip heuristics.
- Phone: YOLOv4-tiny (COCO class: `cell phone`) through OpenCV DNN.

Phone-looking layer:

- Select primary face (largest face box).
- Select primary phone (highest-confidence phone box).
- Consider user "looking at phone" when phone is below the face and horizontally close to it.
- If this condition continues for threshold seconds (default 4.0s), fire a beep alert.

Note: this is still a heuristic for "looking at phone" from webcam geometry.

## Project Structure

```text
FocusGuard/
  README.md
  requirements.txt
  .gitignore
  src/
    app.py
    attention/
      __init__.py
      screen_attention_monitor.py
    detectors/
      __init__.py
      haar_feature_detector.py
      phone_detector.py
    utils/
      __init__.py
      drawing.py
```

## Setup

1. Install dependencies with system Python:

```powershell
py -3.13 -m pip install -r requirements.txt
```

On first run, model files are downloaded automatically to `models/yolo/`.

## Run

Basic run:

```powershell
py -3.13 -m src.app
```

Custom camera and thresholds:

```powershell
py -3.13 -m src.app --camera 0 --width 1280 --height 720 --phone-threshold 4.0 --phone-confidence 0.35
```

## Runtime Display

- Bounding boxes for face, eyes, smile, and lips.
- Phone timer overlay, for example: `Phone timer: 2.8s / 4.0s`.
- State text, for example: `Looking at phone`, `No phone`, `No face`, `Phone too far from face`.
- Alarm when threshold is crossed.

## Tunable Parameters

- `--phone-threshold`: seconds before warning, default `4.0`.
- `--phone-confidence`: detector confidence threshold, default `0.35`.

## Platform Note

- On Windows, alarm uses `winsound.Beep`.
- If beep API is unavailable, the app falls back to terminal bell/message.

## Controls

- Press `q` to quit.

## Practical Tips

- Use good front lighting for stable face/eye detection.
- Keep camera at eye level.
- Increase `--phone-threshold` if alerts fire too quickly.
- Reduce `--phone-confidence` slightly if phone is missed often.
