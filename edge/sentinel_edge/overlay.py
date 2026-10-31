"""Drawn call-outs for the analysed frame: each vehicle gets corner brackets
with its class and track number and, where a plate was found, its native
crop is enlarged beside the vehicle with the current reading (plates with no
complete reading get no call-out). Display only:
nothing here feeds recognition or acceptance, and an enlarged crop adds no
information.
"""

GREEN, AMBER, GREY, WHITE, BLACK = (90, 220, 90), (0, 190, 255), (170, 170, 170), (255, 255, 255), (0, 0, 0)
STATES = {
    'accepted by model — NOT verified': ('ACCEPTED - unverified', GREEN),
    'uncertain': ('UNCERTAIN', AMBER),
    'collecting distinct frames': ('READING', GREY),
}
# Plates in this state get no call-out at all: nothing useful to show on the frame.
HIDDEN_STATES = frozenset({'no complete candidate'})
CLASS_NAMES = {1: 'BICYCLE', 2: 'CAR', 3: 'MOTORCYCLE', 5: 'BUS', 7: 'TRUCK'}
LABEL_TEXT = .6  # Text at 60% of its first size; half was judged too small to read at 720p.


def scale_for(image):
    """Line and text size that stays legible from 480p to 4K."""
    return max(1., image.shape[1] / 1280.)


def thickness_for(scale):
    return max(2, round(2 * scale))


def _banner(view, text, origin, colour, scale):
    import cv2
    size = .55 * scale
    thickness = max(1, round(1.4 * scale))
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
    pad = round(5 * scale)
    x = max(0, min(origin[0], view.shape[1] - width - 2 * pad))
    y = max(0, min(origin[1], view.shape[0] - height - baseline - 2 * pad))
    cv2.rectangle(view, (x, y), (x + width + 2 * pad, y + height + baseline + 2 * pad), colour, -1)
    cv2.putText(view, text, (x + pad, y + pad + height), cv2.FONT_HERSHEY_SIMPLEX, size, BLACK, thickness, cv2.LINE_AA)
    return x, y, width + 2 * pad, height + baseline + 2 * pad


def draw_vehicle(view, track, scale, reading=None):
    """Corner brackets and a tag such as `CAR #5 | ANPR: <reading>`.

    Only the vehicle type and track number are shown; the detector's box score is
    left off the frame.
    """
    import cv2
    x1, y1, x2, y2 = (int(v) for v in track.box)
    thickness = thickness_for(scale)
    arm = int(max(8 * scale, min(x2 - x1, y2 - y1) * .18))
    for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(view, (cx, cy), (cx + dx * arm, cy), GREEN, thickness, cv2.LINE_AA)
        cv2.line(view, (cx, cy), (cx, cy + dy * arm), GREEN, thickness, cv2.LINE_AA)
    label = CLASS_NAMES.get(getattr(track, 'vehicle_class', None), 'VEHICLE')
    label += f' #{track.track_id}'
    if reading:
        label += f' | ANPR: {reading}'
    # Inside the top-right corner, clear of the plate call-out drawn above the left side.
    size = scale * .8 * LABEL_TEXT
    (width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .55 * size, max(1, round(1.4 * size)))
    _banner(view, label, (x2 - width - round(12 * scale), y1 + thickness), GREEN, size)


def draw_plate_callout(view, vehicle_box, plate_box, crop, reading, status, observations, scale,
                       format_adjusted=False):
    """Box the plate and pop its enlarged crop out above the vehicle with the reading.

    `format_adjusted` adds FORMAT-ADJUSTED, UNVERIFIED: the shown reading was made by the
    opt-in format validator swapping look-alike characters, not read as-is by any frame.
    """
    import cv2
    if status in HIDDEN_STATES:
        return
    label, colour = STATES.get(status, (str(status).upper()[:28], GREY))
    if format_adjusted and reading:
        label += ' - FORMAT-ADJUSTED, UNVERIFIED'
    thickness = thickness_for(scale)
    plate = tuple(int(v) for v in plate_box)
    cv2.rectangle(view, plate[:2], plate[2:], colour, thickness)
    height, width = view.shape[:2]
    # Enlarge the crop to a readable size without covering more than a third of the frame.
    target = int(min(max(180 * scale, crop.shape[1] * 1.6), width / 3))
    factor = target / crop.shape[1]
    shown = cv2.resize(crop, (target, max(1, round(crop.shape[0] * factor))), interpolation=cv2.INTER_NEAREST)
    if shown.shape[0] > height / 3:
        factor = (height / 3) / shown.shape[0]
        shown = cv2.resize(shown, (max(1, round(shown.shape[1] * factor)), int(height / 3)), interpolation=cv2.INTER_NEAREST)
    border = max(2, round(3 * scale))
    x = max(border, min(int(vehicle_box[0]), width - shown.shape[1] - border))
    y = int(vehicle_box[1]) - shown.shape[0] - round(60 * scale)
    if y < border:  # No room above the vehicle: place it just inside the top of its box.
        y = min(max(border, int(vehicle_box[1]) + round(34 * scale)), max(border, height - shown.shape[0] - round(60 * scale)))
    view[y:y + shown.shape[0], x:x + shown.shape[1]] = shown[:max(0, height - y), :max(0, width - x)]
    cv2.rectangle(view, (x - border, y - border), (x + shown.shape[1] + border, y + shown.shape[0] + border), colour, border)
    cv2.line(view, (x + shown.shape[1] // 2, y + shown.shape[0] + border),
             ((plate[0] + plate[2]) // 2, plate[1]), colour, max(1, round(scale)), cv2.LINE_AA)
    text = reading if reading else 'no reading yet'
    _, _, _, used = _banner(view, text, (x - border, y + shown.shape[0] + border), WHITE, scale * 1.35 * LABEL_TEXT)
    _banner(view, f'{label} - {observations} frame{"s" if observations != 1 else ""}',
            (x - border, y + shown.shape[0] + border + used), colour, scale * .8 * LABEL_TEXT)
