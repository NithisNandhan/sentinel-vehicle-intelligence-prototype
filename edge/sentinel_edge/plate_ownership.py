"""Conservative native-geometry ownership, before any OCR evidence is added.

Containment cannot prove a plate belongs to a vehicle. It can expose a known
ambiguity: overlapping vehicle regions must not both contribute the same plate
to separate readers. No registration text or model confidence resolves that.
"""
from .plate_continuity import _geometry


def ownership_reason(plate_box, owner_id, vehicle_boxes):
    """Return None only when the proposed owner is the sole plausible region.

    A competing box must contain the plate centre and at least 85 percent of
    its area. Small clipping/rounding at vehicle boundaries is tolerated; a
    partial overlap alone is not enough to reject a separate nearby vehicle.
    Inputs are global native xyxy boxes for the current source timestamp.
    """
    geometry = _geometry(plate_box)
    if geometry is None:
        return 'invalid_plate_box'
    width, height, cx, cy = geometry
    candidates = set()
    for track_id, box in vehicle_boxes:
        if _geometry(box) is None:
            continue
        if not box[0] <= cx <= box[2] or not box[1] <= cy <= box[3]:
            continue
        covered = (max(0., min(plate_box[2], box[2])-max(plate_box[0], box[0])) *
                   max(0., min(plate_box[3], box[3])-max(plate_box[1], box[1])))
        if covered / (width*height) >= .85:
            candidates.add(str(track_id))
    if str(owner_id) not in candidates:
        return 'outside_owner'
    if len(candidates) != 1:
        return 'overlapping_vehicles'
    return None
