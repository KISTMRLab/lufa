import numpy as np

CHANNELS = ["browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight",
            "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight"]


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Embedding must be finite")
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if (norms <= 1e-8).any():
        raise ValueError("Zero embeddings cannot be retrieved")
    return x / norms


def nearest(query, embeddings, top_k=1):
    embeddings = np.asarray(embeddings)
    query = np.asarray(query)
    if embeddings.ndim != 2 or query.ndim != 1 or embeddings.shape[1] != len(query):
        raise ValueError("Expected one query vector matching bank embedding width")
    if not 1 <= top_k <= len(embeddings):
        raise ValueError("top-k must be between1 and bank size")
    scores = normalize(embeddings) @ normalize(query)
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return indices, scores[indices]


def resample(motion, frames):
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 9 or len(motion) < 2 or frames < 2:
        raise ValueError("Need nine-channel motion and at least two frames")
    if not np.isfinite(motion).all() or (motion < 0).any() or (motion > 1).any():
        raise ValueError("Motion must be finite in [0,1]")
    return np.stack([np.interp(np.linspace(0, 1, frames), np.linspace(0, 1, len(motion)), motion[:, c])
                     for c in range(9)], -1).astype(np.float32)
