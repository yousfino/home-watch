"""Reproducible positive-only enrollment with a held-out check before refitting."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .profile import IdentityProfile

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def read_image(path):
    # imdecode supports Windows paths containing Unicode, unlike some imread builds.
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def collect(directory):
    if not directory.is_dir():
        raise ValueError(f'Photo directory does not exist: {directory}')
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f'No supported images found in {directory}')
    return files


def photo_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_indices(count, seed=42):
    if count < 6:
        raise ValueError('Provide at least 6 distinct photos (20 recommended) for enrollment and holdout')
    indices = np.random.default_rng(seed).permutation(count)
    holdout_count = max(1, int(np.ceil(count * .25)))
    return indices[holdout_count:], indices[:holdout_count]


def train(encoder, photos, output, threshold=0.4, seed=42):
    files = collect(photos)
    hashes = [photo_hash(p) for p in files]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate photos found; remove duplicates to avoid holdout leakage')
    train_indices, holdout_indices = split_indices(len(files), seed)
    embeddings = []
    for index, path in enumerate(files):
        print(f'Encoding photo {index+1}/{len(files)}: {path.name}')
        try:
            embeddings.append(encoder.single(read_image(path)))
        except ValueError as error:
            raise ValueError(f'{path.name}: {error}') from error
    provisional = IdentityProfile.fit([embeddings[i] for i in train_indices], threshold)
    results = [{'file': files[i].name, **provisional.predict(embeddings[i])} for i in holdout_indices]
    final = IdentityProfile.fit(embeddings, threshold, hashes)
    report = {
        'method': 'Frozen pretrained ArcFace embeddings; normalized mean identity prototype',
        'seed': seed, 'threshold': threshold, 'threshold_calibrated': False,
        'training_count_for_holdout': len(train_indices), 'holdout_count': len(holdout_indices),
        'holdout_familiar_count': sum(r['label'] == 'Familiar' for r in results),
        'holdout_results': results, 'negative_test_count': 0,
        'holdout_training_hashes': [hashes[i] for i in train_indices],
        'holdout_hashes': [hashes[i] for i in holdout_indices],
        'final_profile_photo_count': len(files),
        'notes': 'Holdout scores were computed before refitting on all photos. Same-session positives only; '
                 'no estimate of false acceptance on other people. Threshold is provisional.',
    }
    final.save(output)
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def evaluate(encoder, profile, familiar, unfamiliar):
    groups = {'Familiar': collect(familiar), 'Unfamiliar': collect(unfamiliar)}
    used = set(profile.image_hashes)
    records = []
    for expected, files in groups.items():
        for path in files:
            digest = photo_hash(path)
            if digest in used:
                raise ValueError(f'{path.name}: repeated or enrolled photo; use independent test images')
            used.add(digest)
            try:
                prediction = profile.predict(encoder.single(read_image(path)))
                records.append({'file': path.name, 'expected': expected, **prediction})
            except ValueError as error:
                records.append({'file': path.name, 'expected': expected, 'error': str(error)})
    classified = [r for r in records if 'label' in r]
    return {'total': len(records), 'classified': len(classified), 'unscorable': len(records)-len(classified),
            'false_accepts': sum(r['expected'] == 'Unfamiliar' and r['label'] == 'Familiar' for r in classified),
            'false_rejects': sum(r['expected'] == 'Familiar' and r['label'] == 'Unfamiliar' for r in classified),
            'correct': sum(r['expected'] == r['label'] for r in classified), 'results': records}
