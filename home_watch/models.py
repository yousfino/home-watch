"""Download pinned OpenCV Zoo models and verify their SHA-256 checksums."""
import hashlib
from pathlib import Path
import urllib.request

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "face_detection_yunet", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "face_recognition_sface_2021dec.onnx": (
        "face_recognition_sface", "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}

def verified(path: Path, checksum: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (folder, checksum) in MODELS.items():
        target = directory / name
        if verified(target, checksum):
            continue
        url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/{folder}/{name}"
        temporary = target.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if not verified(temporary, checksum):
                raise ValueError(f"Checksum failed for {name}")
            temporary.replace(target)
            print(f"Verified {name}")
        finally:
            temporary.unlink(missing_ok=True)
