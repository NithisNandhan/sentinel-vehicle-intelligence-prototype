"""Metadata-only view of a shared recognition session. Never forward media/text."""

STATES = {'reading', 'uncertain', 'model_accepted', 'unreadable', 'paused', 'unavailable', 'completed'}
STATUS_FIELDS = ('camera_id', 'status', 'frame_age_seconds', 'resolution',
                 'vehicle_boxes', 'decoded_frames', 'analysed_frames', 'plate_crops',
                 'unique_model_accepted_unverified', 'model_profile',
                 # An exception class name such as ChangedRecording, never its message.
                 'failure_code')
# Present only for a recording. Names and times, never a path or source URL.
RECORDED_FIELDS = ('session_id', 'recording_name', 'video_position_ms', 'video_duration_ms',
                   'time_basis', 'pacing', 'stale_after_seconds')


def recognition_state(status):
    if status == 'accepted by model — NOT verified':
        return 'model_accepted'
    if status == 'uncertain':
        return 'uncertain'
    if status == 'collecting distinct frames':
        return 'reading'
    return 'unreadable'


def operator_snapshot(snapshot):
    result = {key: snapshot[key] for key in STATUS_FIELDS if key in snapshot}
    # Downloaded footage is labelled as such everywhere; it is never "live".
    recorded = snapshot.get('source_mode') == 'recorded'
    result.update(schema_version='1.0', source_mode='recorded' if recorded else 'live')
    if recorded:
        result.update({key: snapshot[key] for key in RECORDED_FIELDS if key in snapshot})
    tracks = []
    for track in snapshot.get('debug', {}).get('tracks', [])[:32]:
        masked = track.get('masked_plate')
        tracks.append({'track_id': track['track_id'], 'observations': track['observations'],
                       'masked_plate': masked if isinstance(masked, str) and '*' in masked else None,
                       'recognition_state': track['recognition_state'],
                       'candidate_count': len(track.get('candidates', []))})
    result['tracks'] = tracks
    # Vehicles seen in this run, masked text only, for the operator's side list.
    result['vehicles_seen'] = [
        {key: row.get(key) for key in ('key', 'track_id', 'vehicle_class', 'confidence', 'first_ms', 'last_ms',
                                       'observations', 'state', 'masked_plate', 'watchlist')}
        for row in snapshot.get('vehicles_seen', [])[:50] if isinstance(row, dict)]
    status = result.get('status', 'unavailable')
    if status in ('paused', 'unavailable', 'completed'):
        result['recognition_state'] = status
    elif status == 'connecting':
        result['recognition_state'] = 'reading'
    else:
        states = {track['recognition_state'] for track in tracks}
        result['recognition_state'] = next((s for s in ('model_accepted', 'uncertain', 'reading') if s in states), 'unreadable')
    delivery = snapshot.get('delivery')
    if isinstance(delivery, dict):
        result['delivery'] = {'enabled': bool(delivery.get('enabled')), 'outbox_depth': int(delivery.get('outbox_depth', 0))}
    return result
