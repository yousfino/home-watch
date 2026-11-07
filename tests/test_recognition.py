import numpy as np
import pytest
from home_watch.recognition import Thresholds, FaceMatcher
from home_watch.models import verified


def test_decision_bands():
    thresholds = Thresholds()
    assert thresholds.classify(0.7) == "Familiar"
    assert thresholds.classify(0.45) == "Familiar"
    assert thresholds.classify(0.36) == "Unresolved"
    assert thresholds.classify(0.30) == "Unfamiliar"
    assert thresholds.classify(-0.2) == "Unfamiliar"
    assert thresholds.classify(float("nan")) == "Unresolved"


@pytest.mark.parametrize("low,high", [(0.5, 0.3), (0.3, 0.3), (-2, 0.5), (0, 2)])
def test_invalid_thresholds(low, high):
    with pytest.raises(ValueError):
        Thresholds(low, high)


def test_quality_rejects_flat_small_and_clipped_faces():
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    assert not FaceMatcher.usable(image, [20, 20, 100, 100])
    noise = np.random.default_rng(1).integers(0, 256, image.shape, dtype=np.uint8)
    assert FaceMatcher.usable(noise, [20, 20, 100, 100])
    assert not FaceMatcher.usable(noise, [20, 20, 30, 30])
    assert not FaceMatcher.usable(noise, [-5, 20, 100, 100])


def test_checksum_rejects_corruption(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not a model")
    assert not verified(path, "0" * 64)


def test_enrollment_rejects_ambiguous_reference():
    matcher = FaceMatcher.__new__(FaceMatcher)
    matcher.detect = lambda image: [1, 2]
    with pytest.raises(ValueError, match="exactly one"):
        matcher.enroll(None)
