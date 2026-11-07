import json
import cv2
import numpy as np
import pytest
from home_watch.profile import IdentityProfile, normalize
from home_watch.recognition import TEMPLATE, align_face
from home_watch.training import train, split_indices, evaluate
from home_watch.models import verified


def embedding(index=0):
    result = np.zeros(512, dtype=np.float32)
    result[index] = 1
    return result


def test_binary_decision_and_roundtrip(tmp_path):
    profile = IdentityProfile.fit([embedding(), embedding()], .4, ['abc'])
    assert profile.predict(embedding())['label'] == 'Familiar'
    assert profile.predict(embedding(1))['label'] == 'Unfamiliar'
    path = tmp_path / 'profile.npz'
    profile.save(path)
    restored = IdentityProfile.load(path)
    assert restored.predict(embedding()) == profile.predict(embedding())
    assert restored.image_hashes == ['abc']


@pytest.mark.parametrize('value', [np.zeros(512), np.ones(511), np.full(512, np.nan)])
def test_invalid_embeddings(value):
    with pytest.raises(ValueError):
        normalize(value)


@pytest.mark.parametrize('threshold', [-1, 1, float('nan')])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        IdentityProfile.fit([embedding()], threshold)


def test_incompatible_profile(tmp_path):
    path = tmp_path / 'bad.npz'
    np.savez(path, centroid=embedding(), metadata=json.dumps({'model': 'old', 'version': 1}))
    with pytest.raises(ValueError, match='incompatible'):
        IdentityProfile.load(path)


def test_split_is_disjoint_reproducible_and_15_5():
    train_ids, holdout_ids = split_indices(20)
    assert len(train_ids) == 15 and len(holdout_ids) == 5
    assert not set(train_ids) & set(holdout_ids)
    assert set(train_ids) | set(holdout_ids) == set(range(20))
    np.testing.assert_array_equal(holdout_ids, split_indices(20)[1])


def test_alignment_identity_and_scaled_translated_landmarks():
    image = np.random.default_rng(1).integers(0, 256, (112, 112, 3), dtype=np.uint8)
    np.testing.assert_array_equal(align_face(image, TEMPLATE), image)
    large = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    aligned = align_face(large, TEMPLATE * 2)
    np.testing.assert_array_equal(aligned, image)
    with pytest.raises(ValueError, match='Degenerate'):
        align_face(image, np.zeros((5, 2)))


class FakeEncoder:
    def single(self, image):
        return embedding()


def test_training_reports_pre_refit_holdout_and_refits_all(tmp_path):
    photos = tmp_path / 'photos'
    photos.mkdir()
    for i in range(20):
        cv2.imwrite(str(photos / f'{i:02}.png'), np.full((10, 10, 3), i, np.uint8))
    output = tmp_path / 'profile.npz'
    report = train(FakeEncoder(), photos, output)
    assert report['training_count_for_holdout'] == 15
    assert report['holdout_count'] == report['holdout_familiar_count'] == 5
    assert report['negative_test_count'] == 0
    assert report['threshold_calibrated'] is False
    assert not set(report['holdout_training_hashes']) & set(report['holdout_hashes'])
    assert len(IdentityProfile.load(output).image_hashes) == 20
    with pytest.raises(ValueError, match='enrolled photo'):
        evaluate(FakeEncoder(), IdentityProfile.load(output), photos, photos)
    (photos / 'duplicate.png').write_bytes((photos / '00.png').read_bytes())
    with pytest.raises(ValueError, match='Duplicate'):
        train(FakeEncoder(), photos, output)


def test_checksum_rejects_corruption(tmp_path):
    path = tmp_path / 'model.onnx'
    path.write_bytes(b'not a model')
    assert not verified(path, '0' * 64)


def test_evaluation_counts_errors_and_misclassifications(tmp_path):
    familiar = tmp_path / 'familiar'
    unfamiliar = tmp_path / 'unfamiliar'
    familiar.mkdir()
    unfamiliar.mkdir()
    for folder, values in ((familiar, (0, 1)), (unfamiliar, (2, 3))):
        for value in values:
            cv2.imwrite(str(folder / f'{value}.png'), np.full((10, 10, 3), value, np.uint8))

    class Encoder:
        def single(self, image):
            value = int(image[0, 0, 0])
            if value == 3:
                raise ValueError('Expected exactly one face; detected 0')
            return embedding(1) if value == 1 else embedding()

    result = evaluate(Encoder(), IdentityProfile.fit([embedding()]), familiar, unfamiliar)
    assert result['total'] == 4 and result['classified'] == 3
    assert result['unscorable'] == 1
    assert result['false_accepts'] == 1 and result['false_rejects'] == 1
    assert result['correct'] == 1
