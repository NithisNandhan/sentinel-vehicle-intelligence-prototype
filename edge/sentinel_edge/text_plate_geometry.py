"""Text-seeded native four-corner enclosures, independent of OCR answers.

Corners enclose writing; they are never asserted to be physical plate corners.
Rectified copies are used to measure aligned strokes. The native image is never
modified, and returned coordinates always refer to that image.
"""
import math


def _ordered_quad(value, shape):
    import cv2
    import numpy as np
    try:
        points = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    h, w = shape[:2]
    if ((points[:, 0] < 0).any() or (points[:, 0] >= w).any()
            or (points[:, 1] < 0).any() or (points[:, 1] >= h).any()):
        return None
    if not cv2.isContourConvex(points):
        return None
    hull = cv2.convexHull(points).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < 12:
        return None
    # Clockwise image order, starting at the upper-left corner. Plate-like text
    # has its long axis mostly horizontal; extreme vertical regions are rejected.
    centre = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1]-centre[1], hull[:, 0]-centre[0])
    ordered = hull[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    if ordered[1, 0] < ordered[3, 0]:
        ordered = ordered[[0, 3, 2, 1]]
    horizontal = (ordered[1]-ordered[0]+ordered[2]-ordered[3])/2
    vertical = (ordered[3]-ordered[0]+ordered[2]-ordered[1])/2
    width, height = float(np.linalg.norm(horizontal)), float(np.linalg.norm(vertical))
    if width < 12 or height < 5 or not .6 <= width/height <= 12:
        return None
    if abs(math.atan2(float(horizontal[1]), float(horizontal[0]))) > math.radians(60):
        return None
    return ordered


def _transform(points, matrix):
    import cv2
    import numpy as np
    return cv2.perspectiveTransform(np.asarray(points, np.float32).reshape(1, -1, 2), matrix)[0]


def _frame(quad):
    import cv2
    import numpy as np
    width = max(12., (float(np.linalg.norm(quad[1]-quad[0]))+float(np.linalg.norm(quad[2]-quad[3])))/2)
    height = max(5., (float(np.linalg.norm(quad[3]-quad[0]))+float(np.linalg.norm(quad[2]-quad[1])))/2)
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    matrix = cv2.getPerspectiveTransform(quad, target)
    return width, height, matrix


def _compatible(first, second):
    """Return a same-row/two-row relation in the first text polygon's frame."""
    import numpy as np
    width, height, matrix = _frame(first)
    local = _transform(second, matrix)
    lo, hi = local.min(axis=0), local.max(axis=0)
    other_height = hi[1]-lo[1]
    axis = local[1]-local[0]
    if other_height <= 0 or not .5 <= other_height/height <= 1.8:
        return None
    if abs(math.atan2(float(axis[1]), float(axis[0]))) > math.radians(14):
        return None
    shared_y = min(height, hi[1])-max(0., lo[1])
    gap_x = max(0., lo[0]-width, -hi[0])
    if shared_y >= .55*min(height, other_height) and gap_x <= .8*min(height, other_height):
        if (max(width, hi[0])-min(0., lo[0]))/max(height, other_height) <= 12:
            return 'same_row'
    shared_x = min(width, hi[0])-max(0., lo[0])
    gap_y = max(0., lo[1]-height, -hi[1])
    if (shared_x >= .45*min(width, hi[0]-lo[0]) and gap_y <= .85*min(height, other_height)
            and (hi[1] <= .2*height or lo[1] >= .8*height)):
        return 'two_row'
    return None


def _enclosure(polygons):
    import numpy as np
    if len(polygons) == 1:
        return polygons[0].copy()
    _, _, matrix = _frame(polygons[0])
    points = np.concatenate([_transform(p, matrix) for p in polygons])
    lo, hi = points.min(axis=0), points.max(axis=0)
    local = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], np.float32)
    return _transform(local, np.linalg.inv(matrix))


def _overlap_fraction(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = min(max(0., a[2]-a[0])*max(0., a[3]-a[1]), max(0., b[2]-b[0])*max(0., b[3]-b[1]))
    return intersection/area if area else 0.


def _contains_separate_plates(quad, boxes):
    import cv2
    import numpy as np
    contained = []
    for value in boxes:
        try:
            box = tuple(float(v) for v in value)
        except (TypeError, ValueError):
            continue
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            continue
        centre = ((box[0]+box[2])/2, (box[1]+box[3])/2)
        if cv2.pointPolygonTest(np.asarray(quad, np.float32), centre, False) >= 0:
            contained.append(box)
    return any(_overlap_fraction(a, b) < .2 for i, a in enumerate(contained) for b in contained[i+1:])


def _grow_strokes(image, quad, member_quads):
    """Expand along nearby glyph components in an orientation-local copy."""
    import cv2
    import numpy as np
    width, height, matrix = _frame(quad)
    # Keep both work and possible growth bounded even for a large shop banner.
    if width*height > 500_000 or width > 1600 or height > 400:
        return (0., 0., width, height), 0, 0
    member_local = [_transform(p, matrix) for p in member_quads]
    glyph_height = float(np.median([p[:, 1].max()-p[:, 1].min() for p in member_local]))
    extension = min(width*1.5, glyph_height*4)
    vertical = glyph_height*.4
    shift = np.array([[1., 0., extension], [0., 1., vertical], [0., 0., 1.]], np.float32)
    working = cv2.warpPerspective(image, shift@matrix,
        (max(1, math.ceil(width+2*extension)), max(1, math.ceil(height+2*vertical))),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    row_centres = [(float(p[:, 1].min()+p[:, 1].max())/2) for p in member_local]
    alternatives = []
    for mask in (binary, 255-binary):
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        components = []
        for x, y, cw, ch, area in stats[1:]:
            if (.4*glyph_height <= ch <= 1.5*glyph_height and .06 <= cw/ch <= 1.4
                    and area >= 5 and .10 <= area/(cw*ch) <= .98):
                box = (float(x-extension), float(y-vertical), float(x+cw-extension), float(y+ch-vertical))
                if any(abs((box[1]+box[3])/2-centre) <= .5*glyph_height for centre in row_centres):
                    components.append(box)
        alternatives.append(components)
    components = max(alternatives, key=lambda group: sum((b[3]-b[1])**2 for b in group))
    selected = [b for b in components if min(width, b[2]) > max(0., b[0])]
    pending = [b for b in components if b not in selected]
    while selected and pending:
        lo, hi = min(b[0] for b in selected), max(b[2] for b in selected)
        additions = [b for b in pending if min(abs(b[2]-lo), abs(b[0]-hi)) <= .85*glyph_height]
        if not additions:
            break
        selected.extend(additions)
        pending = [b for b in pending if b not in additions]
    if not selected:
        return (0., 0., width, height), 0, 0
    bounds = (min(0., min(b[0] for b in selected)), min(0., min(b[1] for b in selected)),
              max(width, max(b[2] for b in selected)), max(height, max(b[3] for b in selected)))
    added = sum(b[2] < 0 or b[0] > width for b in selected)
    return bounds, len(selected), added


def text_plate_candidates(image, polygons, other_boxes=(), max_candidates=8):
    """Return estimated native text enclosures for separate plate verification.

    Inputs are image-only detector polygons, never recognized strings or labels.
    A weak one-fragment seed remains a candidate even with zero glyph components.
    Known neighbouring plates veto a joined enclosure. Limits are heuristics.
    """
    import cv2
    import numpy as np
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or not image.size):
        raise ValueError('Nonempty native BGR uint8 image required')
    if not isinstance(max_candidates, int) or not 1 <= max_candidates <= 32:
        raise ValueError('Bounded candidate limit required')
    valid = []
    for index, polygon in enumerate(list(polygons)[:128]):
        quad = _ordered_quad(polygon, image.shape)
        if quad is not None:
            if any(float(cv2.intersectConvexConvex(quad, old)[0])/min(cv2.contourArea(quad), cv2.contourArea(old)) >= .98
                   for _, old in valid):
                continue
            valid.append((index, quad))
    seeds = [([index], [quad], 'single_text_region') for index, quad in valid]
    for i, (first_index, first) in enumerate(valid):
        for second_index, second in valid[i+1:]:
            relation = _compatible(first, second)
            if relation:
                seeds.append(([first_index, second_index], [first, second], relation))
    # Plate text can be split into several words. Extend compatible pairs to
    # at most four outlines, while retaining a maximum of two character rows.
    seen_members = {tuple(sorted(members)) for members, _, _ in seeds}
    for members, quads, relation in list(seeds):
        if len(members) != 2 or len(seeds) >= 256:
            continue
        grouped_members, grouped_quads = list(members), list(quads)
        for index, candidate in valid:
            if index in grouped_members or len(grouped_members) >= 4:
                continue
            if not any(_compatible(old, candidate) for old in grouped_quads):
                continue
            proposed = grouped_quads+[candidate]
            enclosed = _enclosure(proposed)
            width, height, matrix = _frame(enclosed)
            local = [_transform(p, matrix) for p in proposed]
            typical_height = float(np.median([p[:, 1].max()-p[:, 1].min() for p in local]))
            centres = sorted(float(p[:, 1].mean()) for p in local)
            rows = []
            for centre in centres:
                if not rows or centre-rows[-1] > .65*typical_height:
                    rows.append(centre)
            if (len(rows) > 2 or height > 3.3*typical_height or width > 12*typical_height
                    or _contains_separate_plates(enclosed, other_boxes)):
                continue
            grouped_members.append(index)
            grouped_quads.append(candidate)
            key = tuple(sorted(grouped_members))
            if key not in seen_members:
                seen_members.add(key)
                seeds.append((sorted(grouped_members), list(grouped_quads),
                              'two_row' if len(rows) == 2 else 'same_row'))
    results = []
    seeds.sort(key=lambda item: (-len(item[0]), item[0]))
    # Output limits also bound component/warp work, not merely final rendering.
    for members, quads, relation in seeds[:max(16, max_candidates*4)]:
        seed = _enclosure(quads)
        if _contains_separate_plates(seed, other_boxes):
            continue
        width, height, matrix = _frame(seed)
        bounds, count, added = _grow_strokes(image, seed, quads)
        glyph_height = min(float(np.linalg.norm(p[3]-p[0])) for p in quads)
        mx, my = max(2., .16*glyph_height), max(2., .25*glyph_height)
        x1, y1, x2, y2 = bounds
        local = np.array([[x1-mx, y1-my], [x2+mx, y1-my], [x2+mx, y2+my], [x1-mx, y2+my]], np.float32)
        quad = _transform(local, np.linalg.inv(matrix))
        quad[:, 0] = np.clip(quad[:, 0], 0, image.shape[1]-1)
        quad[:, 1] = np.clip(quad[:, 1], 0, image.shape[0]-1)
        quad = _ordered_quad(quad, image.shape)
        if quad is None or _contains_separate_plates(quad, other_boxes):
            continue
        box = [max(0, math.floor(float(quad[:, 0].min()))), max(0, math.floor(float(quad[:, 1].min()))),
               min(image.shape[1], math.ceil(float(quad[:, 0].max()))+1),
               min(image.shape[0], math.ceil(float(quad[:, 1].max()))+1)]
        if box[2]-box[0] < 16 or box[3]-box[1] < 7:
            continue
        display_seed=seed.copy()
        display_seed[:,0]=np.clip(display_seed[:,0],0,image.shape[1]-1)
        display_seed[:,1]=np.clip(display_seed[:,1],0,image.shape[0]-1)
        candidate = dict(box=box, quad=quad.astype(float).tolist(), seed_quad=display_seed.astype(float).tolist(),
            text_members=members, character_components=int(count),
            geometry=dict(boundary='estimated_text_enclosure', relation=relation,
                          added_components=int(added), native_pixels_preserved=True,
                          physical_plate_corners_verified=False),
            reasons=['Text-seeded enclosure; physical plate border is not required']+
                    (['Weak stroke evidence; independent verification required'] if count < 3 else []))
        # A larger grouped text hypothesis is useful, but duplicate copies are not.
        duplicate = next((old for old in results if _overlap_fraction(old['box'], box) >= .95
                          and old['text_members'] == members), None)
        if duplicate is None:
            results.append(candidate)
    results.sort(key=lambda item: (-len(item['text_members']), -item['character_components'], item['box']))
    return results[:max_candidates]
