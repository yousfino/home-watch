"""Download pinned pretrained models and verify their SHA-256 checksums."""
import hashlib
from pathlib import Path
import urllib.request

REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODELS = {
    "retinaface_r34.onnx": (
        "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx",
        "bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630"),
    "face_recognition_sface_2021dec.onnx": (
        f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}

def verified(path: Path, checksum: str) -> bool:
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def download(directory: Path, detector_only=False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, checksum) in MODELS.items():
        if detector_only and name != "retinaface_r34.onnx":
            continue
        target = directory / name
        if verified(target, checksum):
            continue
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
