# Home Watch

Milestone 1: detect faces and compare each face with one familiar person's reference photo, using a laptop webcam. Python + pretrained RetinaFace ResNet-34 detection and SFace embeddings run locally on the CPU; no model training, Android app or security camera integration is included yet.

## Windows setup

Install 64-bit Python 3.12 from python.org. Open PowerShell in this repository:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m home_watch download-models
.\.venv\Scripts\python.exe -m pytest -q
```

No virtual environment activation is required. RetinaFace downloads from GitHub Releases (`github.com` and its release asset redirect); SFace downloads from `media.githubusercontent.com`. RetinaFace is pinned to the yakhyo/retinaface-pytorch v0.0.1 release and a SHA-256 digest; SFace is pinned to an OpenCV Zoo revision and its Git LFS SHA-256 digest. Downloads are verified before use. Existing valid SFace weights are reused. The old YuNet file is no longer used. Models are stored in ignored `models/`. The RetinaFace implementation and weights are from [yakhyo/retinaface-pytorch](https://github.com/yakhyo/retinaface-pytorch) (MIT; notice retained in `third_party/retinaface-LICENSE.txt`). SFace model descriptions and license files are in [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models); review the individual model licenses before redistribution.

Create a `data` directory and put a clear, well-lit reference photo at `data/familiar.jpg`. It must contain exactly one face, at least 60 pixels wide and tall, without clipping. Personal images and downloaded models are ignored by Git.

```powershell
.\.venv\Scripts\python.exe -m home_watch compare --reference data/familiar.jpg --image data/test.jpg
.\.venv\Scripts\python.exe -m home_watch webcam --reference data/familiar.jpg
```

Press Q or Escape to stop. Use `--camera 1` if the integrated camera is not camera 0. Enable Windows Settings → Privacy & security → Camera → Let desktop apps access your camera, and close other applications using it. Run webcam mode on the laptop; a cloud task cannot access its camera directly.

## Inspect face detection

RetinaFace ResNet-34 replaces YuNet. It uses aspect-preserving resizing and padding before inference, so large studio photos do not reach the detector at arbitrary resolutions. The default detector input is 640×640 with a confidence cutoff of 0.7. ResNet-34 prioritizes detection accuracy over webcam speed; measured throughput depends on your CPU.

Test detection separately from recognition and save a diagnostic image:

```powershell
.\.venv\Scripts\python.exe -m home_watch detect --image data/familiar.jpg --output data/detected.jpg
```

Open `data/detected.jpg` to inspect boxes and five facial landmarks. The JSON output reports detection confidence and original-image coordinates. This command does not require a reference photo or SFace weights, and does not apply recognition quality filters. To download just RetinaFace, run `python -m home_watch download-models --detector-only` using your virtual environment's Python.

For small or difficult faces, increase detector resolution (slower):

```powershell
.\.venv\Scripts\python.exe -m home_watch detect --image data/familiar.jpg --output data/detected.jpg --detector-size 1280
.\.venv\Scripts\python.exe -m home_watch compare --reference data/familiar.jpg --image data/test.jpg --detector-size 1280
```

All three image/webcam commands accept `--detector-size 640|960|1280` and `--detection-confidence` (default 0.7). Lower confidence may recover weak detections but increases false positives. A detected face can still be unsuitable for reliable recognition; detection and recognition quality are separate checks.

## Results and limitations

Each compared face gets a box, detection confidence, label and cosine similarity (not a probability). No detections produces an empty `faces` list.

- **Familiar:** similarity >= 0.45.
- **Unfamiliar:** similarity <= 0.30.
- **Unresolved:** scores between those thresholds, nonfinite scores, or a face that is too small, blurry or clipped. Quality rejection has a null similarity.

These are initial prototype thresholds, not calibrated security guarantees. Override them using `--familiar-threshold` and `--unfamiliar-threshold` on either command. The Laplacian blur cutoff of 40 and minimum size of 60 are initial heuristics. Poor lighting and pose can still cause incorrect results even after quality filtering. A face detector may also miss a face entirely.

Before accepting milestone 1, try multiple separate photos of the familiar person and consenting other people, then test the webcam at different distances and lighting conditions. Record false matches, false rejections and unresolved cases before tuning thresholds. Using the reference image itself only checks pipeline operation, not recognition accuracy. This prototype has no liveness detection and can accept a photograph; do not use it to grant access or trigger security decisions.

The `detect` command explicitly saves an annotated copy at the chosen output path. Images and embeddings stay in memory during comparison; this application does not save camera frames or upload them. Reference files remain on your disk. Keep photos of consenting participants outside version control.

## Cloud / Linux development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m home_watch download-models
.venv/bin/python -m pytest -q
.venv/bin/python -m home_watch compare --reference data/familiar.jpg --image data/test.jpg
```

Tests cover decision bands, quality rejection, reference ambiguity, integrity checks and RetinaFace anchor layout. With RetinaFace weights available, a blank-image inference check runs too. Real-photo regressions require OpenCV sample photos `lena.jpg`, `messi.jpg` and `david1.jpg` in a directory selected by `HOME_WATCH_TEST_IMAGES`; without those files, the eight photo cases are explicitly skipped.

To reproduce the photo regressions on Linux:

```bash
mkdir -p /tmp/home-watch-samples
curl -fL https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg -o /tmp/home-watch-samples/lena.jpg
curl -fL https://raw.githubusercontent.com/opencv/opencv/master/samples/data/messi5.jpg -o /tmp/home-watch-samples/messi.jpg
curl -fL https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/face/david1.jpg -o /tmp/home-watch-samples/david1.jpg
HOME_WATCH_TEST_IMAGES=/tmp/home-watch-samples .venv/bin/python -m pytest -q
```

Validated in cloud: **18 passed**, including original, 3000×3000, blurred, JPEG quality 10, darkened and 96×96 versions of the Lena sample, two additional face samples, and a blank negative image. These are regression checks, not a recognition benchmark or proof of performance on every poor-quality photo. Full SFace comparison and Windows camera hardware have not been verified here. Image comparison is suitable for cloud validation once model downloads and consented test images are available. Webcam mode requires a local camera and desktop display.
