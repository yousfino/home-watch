"""Face detection, alignment and cosine similarity using CPU inference."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .models import MODELS, verified
from .detection import RetinaFace


@dataclass(frozen=True)
class Thresholds:
    # SFace's example cosine threshold is 0.363. A conservative match band
    # leaves uncertain scores unresolved; calibrate on representative photos.
    unfamiliar: float = 0.30
    familiar: float = 0.45

    def __post_init__(self):
        if not -1 <= self.unfamiliar < self.familiar <= 1:
            raise ValueError("Thresholds must satisfy -1 <= unfamiliar < familiar <= 1")

    def classify(self, score: float) -> str:
        if not np.isfinite(score):
            return "Unresolved"
        if score >= self.familiar:
            return "Familiar"
        if score <= self.unfamiliar:
            return "Unfamiliar"
        return "Unresolved"


class FaceMatcher:
    def __init__(self, directory: Path, thresholds: Thresholds, confidence=0.7, detector_size=640):
        for name, (_, checksum) in MODELS.items():
            if not verified(directory / name, checksum):
                raise ValueError("Missing or invalid models. Run: python -m home_watch download-models")
        names = list(MODELS)
        self.detector = RetinaFace(directory / names[0], confidence, detector_size)
        self.recognizer = cv2.FaceRecognizerSF.create(str(directory / names[1]), "")
        self.thresholds = thresholds

    def detect(self, image):
        return self.detector.detect(image)

    @staticmethod
    def usable(image, face) -> bool:
        x, y, w, h = map(int, face[:4])
        height, width = image.shape[:2]
        if min(w, h) < 60 or x < 0 or y < 0 or x + w > width or y + h > height:
            return False
        crop = cv2.cvtColor(image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(crop, cv2.CV_64F).var() >= 40

    def feature(self, image, face):
        return self.recognizer.feature(self.recognizer.alignCrop(image, face)).copy()

    def enroll(self, image):
        faces = self.detect(image)
        if len(faces) != 1:
            raise ValueError(f"Reference must contain exactly one face; found {len(faces)}. "
                             "Run the detect command to inspect boxes and confidence; "
                             "try --detector-size 1280 for small faces.")
        if not self.usable(image, faces[0]):
            raise ValueError("Reference face is too small, blurry or clipped. Use a clear, well-lit photo.")
        return self.feature(image, faces[0])

    def compare(self, image, reference):
        results = []
        for face in self.detect(image):
            score = None
            label = "Unresolved"
            if self.usable(image, face):
                score = float(self.recognizer.match(reference, self.feature(image, face), cv2.FaceRecognizerSF_FR_COSINE))
                label = self.thresholds.classify(score)
            results.append({"box": [int(v) for v in face[:4]], "label": label, "detection_confidence": float(face[-1]), "similarity": score if score is not None and np.isfinite(score) else None})
        return results
