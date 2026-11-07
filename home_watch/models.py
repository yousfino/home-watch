"""Download pinned pretrained models and verify their SHA-256 checksums."""
import hashlib
from pathlib import Path
import urllib.request

MODELS = {
    "retinaface_r34.onnx": (
        "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_r34.onnx",
        "bd0263dc2a465d32859555cb1741f2d98991eb0053696e8ee33fec583d30e630"),
    "w600k_r50.onnx": (
        "https://github.com/yakhyo/uniface/releases/download/weights/w600k_r50.onnx",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"),
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
