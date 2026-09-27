"""Metadata-only view of a shared recognition session. Never forward media/text."""

STATES = {'reading', 'uncertain', 'model_accepted', 'unreadable', 'paused', 'unavailable'}
STATUS_FIELDS = ('camera_id', 'status', 'frame_age_seconds', 'resolution',
                 'vehicle_boxes', 'decoded_frames', 'analysed_frames', 'plate_crops',
                 'unique_model_accepted_unverified', 'model_profile')


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
    result.update(schema_version='1.0', source_mode='live')
    tracks = []
    for track in snapshot.get('debug', {}).get('tracks', [])[:32]:
        masked = track.get('masked_plate')
        tracks.append({'track_id': track['track_id'], 'observations': track['observations'],
                       'masked_plate': masked if isinstance(masked, str) and '*' in masked else None,
                       'recognition_state': track['recognition_state'],
                       'candidate_count': len(track.get('candidates', []))})
    result['tracks'] = tracks
    status = result.get('status', 'unavailable')
    if status in ('paused', 'unavailable'):
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
