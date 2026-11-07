"""RetinaFace ResNet-34 ONNX inference.

Decoder follows yakhyo/retinaface-pytorch (MIT): strides 8/16/32,
anchor sizes 16..512, box variances 0.1/0.2, BGR mean subtraction.
Outputs use OpenCV FaceRecognizerSF's [xywh, five xy landmarks, score].
"""
import cv2
import numpy as np


def priors(size):
    anchors = []
    for step, sizes in zip((8, 16, 32), ((16, 32), (64, 128), (256, 512))):
        for y in range(size // step):
            for x in range(size // step):
                for length in sizes:
                    anchors.append(((x + .5) * step, (y + .5) * step, length, length))
    return np.asarray(anchors, dtype=np.float32)


class RetinaFace:
    def __init__(self, path, confidence=0.7, size=640):
        if not 0 < confidence < 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        if size not in (640, 960, 1280):
            raise ValueError("Detector size must be 640, 960 or 1280")
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.confidence = confidence
        self.size = size
        self.anchors = priors(size)

    def detect(self, image):
        height, width = image.shape[:2]
        scale = self.size / max(height, width)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(image, (resized_width, resized_height))
        # Padding matches the training mean, making the padded tensor zero.
        canvas = np.full((self.size, self.size, 3), (104, 117, 123), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = resized
        blob = cv2.dnn.blobFromImage(canvas, mean=(104, 117, 123), swapRB=False)
        self.net.setInput(blob)
        loc, conf, landmarks = [a[0] for a in self.net.forward(['loc', 'conf', 'landmarks'])]
        selected = conf[:, 1] >= self.confidence
        if not selected.any():
            return np.empty((0, 15), dtype=np.float32)
        anchor = self.anchors[selected]
        loc, scores, landmarks = loc[selected], conf[selected, 1], landmarks[selected]
        center = anchor[:, :2] + loc[:, :2] * .1 * anchor[:, 2:]
        dimensions = anchor[:, 2:] * np.exp(loc[:, 2:] * .2)
        boxes = np.column_stack((center - dimensions / 2, dimensions))
        points = (anchor[:, None, :2] + landmarks.reshape(-1, 5, 2) * .1 * anchor[:, None, 2:])
        # Undo actual rounded resize factors for boxes and all five landmarks.
        factors = np.array([resized_width / width, resized_height / height], dtype=np.float32)
        boxes /= np.tile(factors, 2)
        points /= factors
        valid = (boxes[:, 0] < width) & (boxes[:, 1] < height) & (boxes[:, 0] + boxes[:, 2] > 0) & (boxes[:, 1] + boxes[:, 3] > 0)
        boxes, scores, points = boxes[valid], scores[valid], points[valid]
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.confidence, 0.4)
        if len(keep) == 0:
            return np.empty((0, 15), dtype=np.float32)
        indices = np.asarray(keep).reshape(-1)
        return np.column_stack((boxes[indices], points[indices].reshape(-1, 10), scores[indices])).astype(np.float32)
