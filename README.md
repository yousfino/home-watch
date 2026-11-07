# Home Watch — step 1: enroll one familiar person

A local, CPU-based face recognition prototype. Train/enroll a personal identity profile from multiple photos, then classify each detected face as **Familiar** or **Unfamiliar**. An image without a detected face returns an empty list. The Android application and security camera integration are future work.

This replaces the old single-reference `compare` workflow. It uses **RetinaFace ResNet-34** to locate faces and **ArcFace ResNet-50** to encode aligned faces. Enrollment fits a normalized mean identity prototype from your photos; the pretrained neural networks remain frozen. Twenty positive photos are useful for this method but insufficient to train a general face recognition network from scratch or learn a calibrated two-class classifier.

## Windows setup

Install 64-bit Python 3.12, then open PowerShell in the repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m home_watch download-models
.\.venv\Scripts\python.exe -m pytest -q
```

If `.venv` already exists, reuse it and run the installation command to add ONNX Runtime. Both pretrained models download from GitHub Releases with SHA-256 verification. SFace and YuNet are no longer used. Weights live in ignored `models/`; the ArcFace model is about 166 MB. No TensorFlow, PyTorch or GPU is required. ONNX Runtime on Windows may require the current Microsoft Visual C++ x64 runtime if importing it reports a DLL error.

## Enroll your photos

Extract `data.zip` so the images are under `data/enrollment/` (nested folders are accepted). Each photo must contain exactly one face of the person being enrolled. Use only that person's photos. The program checks detection counts and exact duplicate files, but cannot independently verify that every image is the intended person.

```powershell
.\.venv\Scripts\python.exe -m home_watch train --photos data/enrollment --output data/identity-profile.npz
```

For 20 photos, this does the following:

1. Detects and aligns the face in every photo, then extracts an ArcFace embedding.
2. Splits the distinct photos reproducibly into 15 enrollment and 5 held-out photos (seed 42).
3. Fits a prototype using only the 15 enrollment photos and reports the held-out predictions.
4. Fits the final identity profile using all 20 photos and saves `data/identity-profile.npz` plus `data/identity-profile.report.json`.

The holdout result belongs to the 15-photo prototype, before the final refit. It does not measure independent performance of the final profile. Photos from the same session are a limited validation set; use new photos from another session for the next check.

The default similarity threshold is **0.4**, chosen before evaluating your photos. It is provisional and not calibrated from negative examples. A score at or above the threshold produces Familiar; a lower score produces Unfamiliar. Scores are cosine similarities, not probabilities. To change the threshold, rerun training with `--threshold VALUE` and evaluate on separate data. Repeatedly tuning against the same holdout makes it a development set, not an independent test set.

## Predict a photo or use your webcam

Use a new photo, outside the enrollment dataset:

```powershell
.\.venv\Scripts\python.exe -m home_watch predict --profile data/identity-profile.npz --image data/new-photo.jpg --output data/prediction.jpg
.\.venv\Scripts\python.exe -m home_watch webcam --profile data/identity-profile.npz
```

Each detected face gets its own box, label, similarity and detection confidence. Press Q or Escape to stop the webcam. For camera problems, enable Windows camera access for desktop apps, close other camera apps, or try `--camera 1`. Webcam mode runs on your laptop, not the cloud environment. ResNet models prioritize accuracy; CPU webcam speed depends on your hardware.

For detection diagnostics or difficult/small faces:

```powershell
.\.venv\Scripts\python.exe -m home_watch detect --image data/new-photo.jpg --output data/detection.jpg --detector-size 1280
```

`train`, `predict`, `webcam`, `evaluate` and `detect` accept `--detector-size 640|960|1280` (default 640) and `--detection-confidence` (default 0.7). Larger input is slower. Enrollment requires exactly one detected face per image; it stops with the failing filename instead of silently discarding it. The old hardcoded blur rejection is removed.

## Evaluate unfamiliar-person rejection

Collect **new photos of yourself** in `data/test/familiar/` and **photos of consenting other people** in `data/test/unfamiliar/`. Neither folder should contain enrolled photos. Each test image should contain exactly one face.

```powershell
.\.venv\Scripts\python.exe -m home_watch evaluate --profile data/identity-profile.npz --familiar data/test/familiar --unfamiliar data/test/unfamiliar --output data/evaluation.json
```

The report gives correct classifications, false accepts, false rejects, and unscorable images separately. No-face and multiple-face images are unscorable during evaluation, rather than being counted as successful rejections. Exact duplicates and images used for enrollment are rejected to prevent obvious test leakage; resized or re-encoded copies cannot be detected by this check.

Only testing photos of yourself cannot establish whether other people will be rejected. Test multiple people and real webcam conditions before relying on the threshold. This prototype has no liveness detection and can match a photograph of the enrolled person.

## Data and model files

Keep your photos, identity profile and reports private. `data/`, `models/`, `.npz` files and ZIP archives are ignored by Git. No personal photos or embeddings belong in commits. The profile is biometric data even though it is not a photograph. Inference runs locally and does not upload images. Model downloads use the network. Images are saved only when an explicit output path is supplied to `detect` or `predict`; webcam frames are not saved.

RetinaFace weights and decoder reference: [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch), MIT notice retained in `third_party/retinaface-LICENSE.txt`. ArcFace `w600k_r50.onnx` originates from InsightFace and is downloaded from [UniFace's model release](https://github.com/yakhyo/uniface/releases/tag/weights), with a pinned digest. InsightFace's [pretrained-model policy](https://github.com/deepinsight/insightface#license) limits these weights to non-commercial research; a commercial camera product needs appropriate model licensing or different weights.

## Development checks

On Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`. Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

Tests cover profile persistence/model compatibility, binary decisions, invalid embeddings, deterministic disjoint holdouts, alignment, duplicate leakage, evaluation and detector geometry. Real detector tests run when weights are present; photo regression cases need OpenCV samples selected via `HOME_WATCH_TEST_IMAGES`. Without the samples, those cases are explicitly skipped.

```bash
mkdir -p /tmp/home-watch-samples
curl -fL https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg -o /tmp/home-watch-samples/lena.jpg
curl -fL https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg -o /tmp/home-watch-samples/messi.jpg
curl -fL https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/face/david1.jpg -o /tmp/home-watch-samples/david1.jpg
HOME_WATCH_TEST_IMAGES=/tmp/home-watch-samples .venv/bin/python -m pytest -q
```

Personal training data and identity reports are excluded from these source-controlled tests. Windows camera behavior requires testing on the laptop.
