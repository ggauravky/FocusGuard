# FocusGuard (OpenCV + Python)

A structured Python project for real-time detection of:

- Face
- Eyes
- Smile
- Lips

The app uses OpenCV Haar cascades for face/eyes/smile and a color-based lip region detector inside the lower face ROI.

## Project Structure

```text
FocusGuard/
  README.md
  requirements.txt
  .gitignore
  src/
    app.py
    detectors/
      __init__.py
      haar_feature_detector.py
    utils/
      __init__.py
      drawing.py
```

## Setup

1. Install dependencies (system Python):

```powershell
py -3.13 -m pip install -r requirements.txt
```

## Run

```powershell
py -3.13 -m src.app
```

Optional camera config:

```powershell
py -3.13 -m src.app --camera 0 --width 1280 --height 720
```

## Controls

- Press `q` to quit.

## Notes

- Detection quality depends on lighting and camera quality.
- Smile/lip detection can be tuned in `src/detectors/haar_feature_detector.py`.
