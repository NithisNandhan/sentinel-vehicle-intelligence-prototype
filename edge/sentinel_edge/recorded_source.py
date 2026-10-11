"""Permitted local recordings as a source for the shared recognition session.

A file is listed only when a permission receipt sits beside it. Decoding keeps
the file's own presentation timestamps; nothing here invents a time. Frames are
decoded to memory and never re-written to disk. Recorded footage is always
labelled `recorded`, never `live`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import time

from .stream import VideoFrame

VIDEO_SUFFIXES = frozenset({'.mp4', '.mkv', '.mov', '.avi', '.m4v', '.ts', '.webm', '.ogv', '.ogg'})
RECEIPT_SUFFIX = '.receipt.json'
REQUIRED_RECEIPT_TEXT = ('source', 'licence', 'permission')
_SHA256 = re.compile(r'^[a-f0-9]{64}$')
_HASH_CACHE: dict = {}


class RecordingError(ValueError):
    """Class names are the only detail published; messages never carry paths."""
    @property
    def failure_code(self):
        return type(self).__name__


class UnpermittedRecording(RecordingError): pass
class ChangedRecording(RecordingError): pass
class CorruptRecording(RecordingError): pass
class MissingPts(RecordingError): pass
class NoVideoStream(RecordingError): pass
class InvalidSeek(RecordingError): pass


@dataclass(frozen=True)
class Recording:
    recording_id: str
    name: str
    path: Path
    sha256: str
    duration_ms: int | None = None
    start_pts_ms: int = 0
    started_at: datetime | None = None  # Only when the receipt states it.

    # The shared session addresses sources by these two attributes.
    @property
    def camera_id(self):
        return self.recording_id

    def public(self):
        return {'id': self.recording_id, 'name': self.name, 'duration_ms': self.duration_ms,
                'time_basis': 'source_clock' if self.started_at else 'processing_time'}


def file_sha256(path):
    """Hash once per unchanged file; recordings can be several gigabytes."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if len(_HASH_CACHE) > 256:
            _HASH_CACHE.clear()
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def _display_name(value, fallback):
    # Display text only: no control characters and nothing that looks like a URL.
    if not isinstance(value, str) or not value.strip() or len(value) > 120 or '://' in value \
            or any(ord(c) < 32 for c in value):
        return fallback[:120]
    return value.strip()


def _utc(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise UnpermittedRecording('Recording start time must be an ISO 8601 UTC string or null')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise UnpermittedRecording('Recording start time is not ISO 8601') from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise UnpermittedRecording('Recording start time must be UTC')
    return parsed


def read_receipt(video):
    """Return (sha256, title, started_at) from the receipt beside a video."""
    receipt = Path(str(video) + RECEIPT_SUFFIX)
    try:
        data = json.loads(receipt.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise UnpermittedRecording('No readable permission receipt') from None
    if not isinstance(data, dict) or data.get('schema_version') != '1.0':
        raise UnpermittedRecording('Unsupported permission receipt')
    if any(not isinstance(data.get(key), str) or not data[key].strip() for key in REQUIRED_RECEIPT_TEXT):
        raise UnpermittedRecording('Receipt must state source, licence and permission')
    digest = data.get('sha256')
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise UnpermittedRecording('Receipt must state the file SHA256')
    return digest, _display_name(data.get('title'), Path(video).stem), _utc(data.get('recorded_start_utc'))


def write_receipt(video, *, source, licence, permission, title=None, recorded_start_utc=None,
                  obtained_on=None):
    """Create the receipt for a file the caller has permission to use. Never overwrites."""
    video = Path(video)
    if video.suffix.lower() not in VIDEO_SUFFIXES or not video.is_file():
        raise UnpermittedRecording('Not a supported video file')
    data = {'schema_version': '1.0', 'sha256': file_sha256(video), 'title': title or video.stem,
            'source': source, 'licence': licence, 'permission': permission,
            'obtained_on': obtained_on, 'recorded_start_utc': recorded_start_utc}
    if any(not isinstance(data[key], str) or not data[key].strip() for key in REQUIRED_RECEIPT_TEXT):
        raise UnpermittedRecording('Receipt must state source, licence and permission')
    _utc(recorded_start_utc)
    receipt = Path(str(video) + RECEIPT_SUFFIX)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
    return receipt


def _probe(path):
    """Duration and first timestamp from container metadata, if it states them."""
    import av
    try:
        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == 'video'), None)
            if stream is None:
                raise NoVideoStream('No video stream')
            base = float(stream.time_base) if stream.time_base else None
            start = round(stream.start_time * base * 1000) if base and stream.start_time is not None else 0
            if base and stream.duration:
                duration = round(stream.duration * base * 1000)
            elif container.duration:
                duration = round(container.duration / 1000)  # av.time_base is microseconds
            else:
                duration = None
            return duration, start
    except av.error.FFmpegError:
        raise CorruptRecording('Recording could not be opened') from None


class RecordingLibrary:
    """Videos directly inside one folder that each have a valid receipt."""
    def __init__(self, root):
        self.root = Path(root)
        self._probed = {}

    def scan(self):
        """Return ({id: Recording}, [reasons for files that were not listed])."""
        found, rejected = {}, []
        if not self.root.is_dir():
            return found, rejected
        for path in sorted(self.root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                digest, title, started_at = read_receipt(path)
                stat = path.stat()
                key = (path.name, stat.st_size, stat.st_mtime_ns)
                if key not in self._probed:
                    self._probed = {k: v for k, v in self._probed.items() if k[0] != path.name}
                    self._probed[key] = _probe(path)
                duration, start = self._probed[key]
            except RecordingError as error:
                rejected.append({'file': path.name, 'reason': error.failure_code})
                continue
            recording_id = 'rec-' + digest[:12]
            if recording_id in found:
                rejected.append({'file': path.name, 'reason': 'DuplicateRecording'})
                continue
            found[recording_id] = Recording(recording_id, title, path, digest, duration, start, started_at)
        return found, rejected


@dataclass(frozen=True)
class RecordedRun:
    """One pass over a recording. A restart or seek is a new run and session."""
    recording: Recording
    session_id: str
    start_ms: int = 0
    paced: bool = True
    paused: object | None = None

    def frames(self, stop):
        return read_recording(self.recording, stop, paced=self.paced, start_ms=self.start_ms,
                              paused=self.paused)

    def observed_at(self, pts_ms):
        """(UTC time, basis). The recording's own time only if its receipt states it."""
        if self.recording.started_at is None:
            return datetime.now(timezone.utc), 'processing_time'
        offset = timedelta(milliseconds=pts_ms - self.recording.start_pts_ms)
        return self.recording.started_at + offset, 'source_clock'


def new_session_id():
    """One per start, restart or seek, so a replay is traceable as a new run."""
    return 's-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(4)


def read_recording(recording, stop, *, paced=True, start_ms=0, paused=None, verify=True,
                   clock=time.monotonic):
    """Yield ('file', VideoFrame) in presentation order until the file ends.

    `paced` waits so playback is no faster than real time; it never drops a
    frame. `start_ms` is an offset from the start of the video. A frame with no
    presentation timestamp, or a file that fails to decode, raises instead of
    being skipped. `paused` is an Event-like object; while set, decoding waits.
    """
    import av
    if isinstance(start_ms, bool) or not isinstance(start_ms, int) or start_ms < 0 or (
            recording.duration_ms is not None and start_ms > recording.duration_ms):
        raise InvalidSeek('Seek position is outside the recording')
    if verify and file_sha256(recording.path) != recording.sha256:
        raise ChangedRecording('Recording no longer matches its receipt')
    wait = getattr(stop, 'wait', None) or (lambda seconds: time.sleep(seconds) or False)
    try:
        with av.open(str(recording.path)) as container:
            stream = next((s for s in container.streams if s.type == 'video'), None)
            if stream is None:
                raise NoVideoStream('No video stream')
            target = recording.start_pts_ms + start_ms
            if start_ms:
                container.seek(round(target / 1000 / float(stream.time_base)), stream=stream,
                               backward=True, any_frame=False)
            previous = origin_pts = origin_clock = None
            for frame in container.decode(stream):
                if stop.is_set():
                    return
                if frame.pts is None:
                    raise MissingPts('Frame has no presentation timestamp')
                pts_ms = round(float(frame.pts * frame.time_base) * 1000)
                if pts_ms < target:
                    continue  # Decoding restarts at the keyframe before a seek.
                if paused is not None and paused.is_set():
                    began = clock()
                    while paused.is_set() and not stop.is_set():
                        wait(.1)
                    if origin_clock is not None:
                        origin_clock += clock() - began
                    if stop.is_set():
                        return
                discontinuity = previous is not None and (pts_ms < previous or pts_ms - previous > 5000)
                if origin_pts is None or discontinuity:
                    origin_pts, origin_clock = pts_ms, clock()
                elif paced:
                    remaining = origin_clock + (pts_ms - origin_pts) / 1000 - clock()
                    while remaining > 0 and not stop.is_set():
                        wait(min(remaining, .25))
                        remaining = origin_clock + (pts_ms - origin_pts) / 1000 - clock()
                    if stop.is_set():
                        return
                previous = pts_ms
                yield 'file', VideoFrame(frame.to_ndarray(format='bgr24'), pts_ms, discontinuity)
    except av.error.FFmpegError:
        raise CorruptRecording('Recording could not be decoded') from None
