"""RetinaFace detection + ArcFace ResNet-50 embeddings on the CPU."""
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
from .models import MODELS, verified
from .detection import RetinaFace

# Standard ArcFace 112x112 alignment template; points are in image-left to
# image-right order: eyes, nose, mouth corners (matching RetinaFace output).
TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def align_face(image, landmarks):
    """Least-squares similarity transform using all five landmarks (Umeyama)."""
    source = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    target = TEMPLATE.astype(np.float64)
    if not np.isfinite(source).all():
        raise ValueError('Nonfinite facial landmarks')
    src_mean, dst_mean = source.mean(axis=0), target.mean(axis=0)
    src, dst = source - src_mean, target - dst_mean
    variance = np.sum(src ** 2) / len(src)
    if variance < 1e-8:
        raise ValueError('Degenerate facial landmarks')
    u, singular, vt = np.linalg.svd(dst.T @ src / len(src))
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    scale = np.sum(singular * np.diag(correction)) / variance
    matrix = np.column_stack((scale * rotation, dst_mean - scale * rotation @ src_mean))
    return cv2.warpAffine(image, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class FaceEncoder:
    def __init__(self, directory: Path, confidence=0.7, detector_size=640):
        for name, (_, checksum) in MODELS.items():
            if not verified(directory / name, checksum):
                raise ValueError('Missing or invalid pretrained models. Run: python -m home_watch download-models')
        self.detector = RetinaFace(directory / 'retinaface_r34.onnx', confidence, detector_size)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.recognizer = ort.InferenceSession(str(directory / 'w600k_r50.onnx'),
                                               sess_options=options, providers=['CPUExecutionProvider'])
        self.input_name = self.recognizer.get_inputs()[0].name

    def detect(self, image):
        return self.detector.detect(image)

    def feature(self, image, face):
        aligned = align_face(image, face[4:14])
        tensor = cv2.dnn.blobFromImage(aligned, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        return self.recognizer.run(None, {self.input_name: tensor})[0].reshape(-1)

    def single(self, image):
        faces = self.detect(image)
        if len(faces) != 1:
            raise ValueError(f'Expected exactly one face; detected {len(faces)}')
        return self.feature(image, faces[0])

    def predict(self, image, profile):
        return [{**profile.predict(self.feature(image, face)), 'box': face[:4].astype(int).tolist(),
                 'detection_confidence': float(face[-1])} for face in self.detect(image)]
