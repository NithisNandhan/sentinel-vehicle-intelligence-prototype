"""Small learned pre-OCR visual gate; no recognizer, plate text or pickle.

Scores are classifier outputs on a limited annotation-derived development set,
not calibrated probabilities of real CCTV plate identity or reading correctness.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np


FEATURE_VERSION = 'native-gray-numpyhog-128x64-v1'
FEATURE_DIM = 3786


def visual_hog(gray):
    """Fixed unsigned nine-bin HOG; independent of OpenCV objdetect packaging."""
    gx = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=1)
    gy = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=1)
    magnitude = np.hypot(gx,gy).ravel()
    orientation = (np.rad2deg(np.arctan2(gy,gx))%180/20).ravel()
    lower = np.floor(orientation).astype(np.int32)%9
    fraction = orientation-np.floor(orientation)
    yy,xx = np.indices(gray.shape)
    cells = ((yy//8)*16+xx//8).ravel()
    histogram = np.zeros((128,9),np.float32)
    np.add.at(histogram,(cells,lower),magnitude*(1-fraction))
    np.add.at(histogram,(cells,(lower+1)%9),magnitude*fraction)
    histogram = histogram.reshape(8,16,9)
    blocks=np.lib.stride_tricks.sliding_window_view(histogram,(2,2),axis=(0,1))
    blocks=blocks.transpose(0,1,3,4,2).reshape(7,15,36).copy()
    blocks/=np.sqrt((blocks**2).sum(axis=2,keepdims=True)+1e-6)
    np.minimum(blocks,.2,out=blocks)
    blocks/=np.sqrt((blocks**2).sum(axis=2,keepdims=True)+1e-6)
    return blocks.ravel()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_features(image):
    """Original pixels become a padded working view; caller input is unchanged."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size or min(image.shape[:2]) < 4):
        raise ValueError('Nonempty uint8 BGR region at least four pixels wide/high required')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(128/w, 64/h)
    rw, rh = max(1, round(w*scale)), max(1, round(h*scale))
    working = np.full((64, 128), round(float(np.median(gray))), np.uint8)
    resized = cv2.resize(gray, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    left, top = (128-rw)//2, (64-rh)//2
    working[top:top+rh, left:left+rw] = resized
    hog = visual_hog(working)
    dx = np.abs(np.diff(resized.astype(np.float32), axis=1)).mean() if rw > 1 else 0.
    dy = np.abs(np.diff(resized.astype(np.float32), axis=0)).mean() if rh > 1 else 0.
    texture = np.array([min(w/h, 8.)/8., gray.std()/128., gray.mean()/255.,
                        np.mean(cv2.Canny(resized, 50, 150) > 0), dx/64., dy/64.], np.float32)
    features = np.concatenate((hog, texture)).astype(np.float32)
    if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
        raise ValueError('Invalid visual feature vector')
    return features


def decision(score, reject_below, accept_above):
    if (not np.isfinite([score, reject_below, accept_above]).all()
            or not 0 <= reject_below < accept_above <= 1 or not 0 <= score <= 1):
        raise ValueError('Finite classifier score and ordered thresholds required')
    if score < reject_below:
        return 'likely_non_plate'
    if score >= accept_above:
        return 'likely_plate'
    return 'uncertain'


class PlateRegionVerifier:
    def __init__(self, model_path):
        self.path = Path(model_path)
        receipt_path = self.path.with_suffix('.json')
        if self.path.stat().st_size > 2*1024**2 or receipt_path.stat().st_size > 256*1024:
            raise ValueError('Verifier artifact exceeds bounded size')
        self.receipt = json.loads(receipt_path.read_bytes())
        self.model_sha256 = sha256(self.path)
        if (self.receipt.get('model_sha256') != self.model_sha256
                or self.receipt.get('feature_version') != FEATURE_VERSION
                or self.receipt.get('feature_dimension') != FEATURE_DIM
                or self.receipt.get('model_type') != 'linear-logistic-hog'):
            raise ValueError('Verifier artifact identity or feature contract mismatch')
        with zipfile.ZipFile(self.path) as archive:
            if (set(archive.namelist()) != {'weights.npy','bias.npy'}
                    or sum(info.file_size for info in archive.infolist()) > FEATURE_DIM*8+4096):
                raise ValueError('Verifier array archive exceeds expected feature size')
        with np.load(self.path, allow_pickle=False) as data:
            if set(data.files) != {'weights', 'bias'}:
                raise ValueError('Unexpected verifier arrays')
            self.weights = data['weights'].astype(np.float32)
            self.bias = data['bias'].astype(np.float64)
        if self.weights.shape != (FEATURE_DIM,) or self.bias.shape != (1,) or not np.isfinite(self.weights).all() or not np.isfinite(self.bias).all():
            raise ValueError('Malformed verifier weights')
        thresholds = self.receipt['thresholds']
        self.reject_below = float(thresholds['reject_below'])
        self.accept_above = float(thresholds['accept_above'])
        decision(.5, self.reject_below, self.accept_above)

    def verify(self, image, context=None):
        features = extract_features(image)
        logit = float(features@self.weights+float(self.bias[0]))
        score = float(1/(1+np.exp(-np.clip(logit, -60, 60))))
        status = decision(score, self.reject_below, self.accept_above)
        reason = {'likely_plate': 'Visual classifier exceeds the validation-selected acceptance threshold',
                  'likely_non_plate': 'Visual classifier is below the conservative validation rejection threshold',
                  'uncertain': 'Visual evidence lies between the rejection and acceptance thresholds'}[status]
        return dict(status=status, classifier_score=score, reasons=[reason],
                    model_sha256=self.model_sha256, allow_ocr=status == 'likely_plate',
                    scope='Visual candidate triage; uncalibrated score, not plate identity or OCR confidence',
                    training_label_scope=self.receipt.get('label_scope'))
