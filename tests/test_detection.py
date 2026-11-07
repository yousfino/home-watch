"""Geometry and real-model regression tests (no personal photos required)."""
import os
from pathlib import Path
import cv2
import numpy as np
import pytest
from home_watch.detection import RetinaFace, priors
from home_watch.models import MODELS, verified


def test_anchor_layout():
    anchors = priors(640)
    assert anchors.shape == (16800, 4)
    np.testing.assert_array_equal(anchors[0], [4, 4, 16, 16])
    np.testing.assert_array_equal(anchors[1], [4, 4, 32, 32])
    np.testing.assert_array_equal(anchors[12800], [8, 8, 64, 64])


@pytest.fixture(scope="module")
def detector():
    path = Path("models/retinaface_r34.onnx")
    if not verified(path, MODELS[path.name][1]):
        pytest.skip("Download the RetinaFace model to run real inference tests")
    return RetinaFace(path)


def test_blank_has_no_faces(detector):
    assert len(detector.detect(np.full((480, 640, 3), 255, dtype=np.uint8))) == 0


@pytest.mark.parametrize("variant", ["original", "large", "blurred", "jpeg", "dark", "small"])
def test_real_face_variants(detector, variant):
    samples = os.environ.get("HOME_WATCH_TEST_IMAGES")
    if not samples:
        pytest.skip("Set HOME_WATCH_TEST_IMAGES to OpenCV sample image directory")
    image = cv2.imread(str(Path(samples) / "lena.jpg"))
    assert image is not None
    if variant == "large":
        image = cv2.resize(image, (3000, 3000))
    elif variant == "blurred":
        image = cv2.GaussianBlur(image, (15, 15), 4)
    elif variant == "jpeg":
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 10])
        assert ok
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    elif variant == "dark":
        image = (image * .3).astype(np.uint8)
    elif variant == "small":
        image = cv2.resize(image, (96, 96))
    faces = detector.detect(image)
    assert len(faces) == 1, (variant, faces)
    x, y, w, h = faces[0, :4] / np.tile([image.shape[1], image.shape[0]], 2)
    assert .3 < x < .6 and .25 < y < .6
    assert .15 < w < .45 and .2 < h < .6
    assert np.isfinite(faces).all()


@pytest.mark.parametrize("name", ["messi.jpg", "david1.jpg"])
def test_other_real_faces(detector, name):
    samples = os.environ.get("HOME_WATCH_TEST_IMAGES")
    if not samples:
        pytest.skip("Set HOME_WATCH_TEST_IMAGES to OpenCV sample image directory")
    image = cv2.imread(str(Path(samples) / name))
    assert image is not None
    assert len(detector.detect(image)) >= 1
