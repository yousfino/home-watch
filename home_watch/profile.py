"""A positive-only identity prototype; no neural-network weights are retrained."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

MODEL_ID = 'arcface-w600k-r50-4c06341c'


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.shape != (512,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Expected a finite, nonzero 512-dimensional embedding')
    return vector / norm


@dataclass
class IdentityProfile:
    centroid: np.ndarray
    threshold: float
    image_hashes: list[str]

    def __post_init__(self):
        self.centroid = normalize(self.centroid)
        if not np.isfinite(self.threshold) or not -1 < self.threshold < 1:
            raise ValueError('Threshold must be finite and between -1 and 1')

    @classmethod
    def fit(cls, embeddings, threshold=0.4, image_hashes=None):
        if len(embeddings) == 0:
            raise ValueError('No embeddings to enroll')
        centroid = np.mean(np.stack([normalize(e) for e in embeddings]), axis=0)
        return cls(centroid, threshold, list(image_hashes or []))

    def predict(self, embedding):
        score = float(np.clip(self.centroid @ normalize(embedding), -1, 1))
        return {'label': 'Familiar' if score >= self.threshold else 'Unfamiliar', 'similarity': score}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps({'version': 1, 'model': MODEL_ID, 'threshold': self.threshold,
                               'image_hashes': self.image_hashes})
        with path.open('wb') as stream:
            np.savez_compressed(stream, centroid=self.centroid, metadata=metadata)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata.get('version') != 1 or metadata.get('model') != MODEL_ID:
                raise ValueError('Profile was made with an incompatible recognition model')
            return cls(data['centroid'], metadata['threshold'], metadata['image_hashes'])
