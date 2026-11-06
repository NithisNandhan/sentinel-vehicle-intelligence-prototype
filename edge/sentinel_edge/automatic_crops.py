"""Image-only plate proposals at native resolution, isolated from OCR/delivery.

All selection uses model proposals and image geometry. Manual annotations are
deliberately absent from this interface. Native crops are unresized evidence;
optional rectification is a separately labelled, interpolated working view.
"""
from dataclasses import dataclass
import math


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def clip(box, shape):
    if box is None or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        return None
    h, w = shape[:2]
    result = (max(0, math.floor(box[0])), max(0, math.floor(box[1])),
              min(w, math.ceil(box[2])), min(h, math.ceil(box[3])))
    return result if result[2]-result[0] >= 24 and result[3]-result[1] >= 8 else None


def windows(shape, size, fraction=.25):
    """Overlapping windows cover the entire original without resizing it."""
    if not 256 <= size <= 2048 or not 0 < fraction <= .5:
        raise ValueError('Invalid native tiling')
    h, w = shape[:2]
    def starts(length):
        end = max(0, length-size)
        return sorted(set([*range(0, end+1, max(1, round(size*(1-fraction)))), end]))
    return [(x, y, min(x+size,w), min(y+size,h)) for y in starts(h) for x in starts(w)]


@dataclass(frozen=True)
class Proposal:
    box: tuple
    source: str
    border_cut: bool = False
    rank: int = 0


class NativePlateCropper:
    def __init__(self, detector, *, sizes=(1280, 768), max_calls=160, max_proposals=128):
        if not sizes or len(sizes) > 3 or any(not 256 <= n <= 2048 for n in sizes):
            raise ValueError('Bounded tile sizes required')
        if not 1 <= max_calls <= 256 or not 1 <= max_proposals <= 256:
            raise ValueError('Bounded crop work required')
        self.detector, self.sizes = detector, sizes
        self.max_calls, self.max_proposals = max_calls, max_proposals
        self.calls = 0
        self.budget_exhausted = False

    def _detect(self, image):
        if self.calls >= self.max_calls:
            self.budget_exhausted = True
            return []
        self.calls += 1
        return self.detector.boxes(image)

    def proposals(self, image):
        self.calls, self.budget_exhausted = 0, False
        h, w = image.shape[:2]
        # Detect small plates directly: a missed vehicle cannot veto a plate.
        pending = []
        scans = [('full', (0,0,w,h))]
        seen = {(0,0,w,h)}
        for size in self.sizes:
            for window in windows(image.shape, size):
                if window not in seen:
                    scans.append((f'tile-{size}',window))
                    seen.add(window)
        for source, (x,y,x2,y2) in scans:
            tile = image[y:y2,x:x2]
            for rank, raw in enumerate(self._detect(tile)):
                local = clip(raw,tile.shape)
                if local is None:
                    continue
                a,b,c,d = local
                # Flag artificial tile cuts, not genuine image borders.
                cut = ((a <= 2 and x > 0) or (b <= 2 and y > 0)
                       or (c >= x2-x-2 and x2 < w) or (d >= y2-y-2 and y2 < h))
                pending.append(Proposal((a+x,b+y,c+x,d+y),source,cut,rank))
        # Confidence order is available only within each detector call. Do not
        # invent cross-tile model scores. Prefer intact boxes, then local rank.
        selected = []
        for item in sorted(pending,key=lambda p:(p.border_cut,p.rank)):
            if any(overlap(item.box,old.box) >= .45 for old in selected):
                continue
            selected.append(item)
            if len(selected) >= self.max_proposals:
                break
        return selected

    def refine(self, image, proposal):
        """One second look with context; it must still describe the same plate."""
        x,y,x2,y2 = proposal.box
        w,h = x2-x,y2-y
        region = (max(0,x-w//2),max(0,y-h),min(image.shape[1],x2+w//2),min(image.shape[0],y2+h))
        a,b,c,d = region
        candidates = []
        for raw in self._detect(image[b:d,a:c]):
            box = clip(raw,(d-b,c-a,3))
            if box is None:
                continue
            box = (box[0]+a,box[1]+b,box[2]+a,box[3]+b)
            cx,cy = (box[0]+box[2])/2,(box[1]+box[3])/2
            if x <= cx <= x2 and y <= cy <= y2 and overlap(box,proposal.box) >= .4:
                candidates.append(box)
        best = max(candidates,key=lambda box:overlap(box,proposal.box),default=None)
        return Proposal(best,proposal.source+'+local',False,proposal.rank) if best else proposal


def native_crop(image, box, *, margin=.04):
    if not 0 <= margin <= .2:
        raise ValueError('Small crop margin required')
    valid = clip(box,image.shape)
    if valid is None:
        raise ValueError('Invalid plate box')
    x,y,x2,y2 = valid
    px,py = max(1,math.ceil((x2-x)*margin)),max(1,math.ceil((y2-y)*margin))
    padded = (max(0,x-px),max(0,y-py),min(image.shape[1],x2+px),min(image.shape[0],y2+py))
    a,b,c,d = padded
    return image[b:d,a:c].copy(),padded


def rectify_boundary(image, box):
    """Optional visible quadrilateral; uncertain contours leave the crop intact.

    This is geometric rectification, not generated or recovered detail. Original
    pixels are always retained separately. A plate-corner model is not implied.
    """
    import cv2
    import numpy as np
    crop, outer = native_crop(image,box,margin=.15)
    ox,oy,_,_ = outer
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    area = (box[2]-box[0])*(box[3]-box[1])
    choices = []
    for contour in contours:
        quad = cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        points = quad[:,0,:].astype(np.float32)
        if not .45*area <= cv2.contourArea(points) <= 1.6*area:
            continue
        center = points.mean(axis=0)
        points = points[np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))]
        points = np.roll(points,-np.argmin(points.sum(axis=1)),axis=0)
        a,b,c,d = points
        width = max(np.linalg.norm(b-a),np.linalg.norm(c-d))
        height = max(np.linalg.norm(d-a),np.linalg.norm(c-b))
        if not 1.2 <= width/max(1,height) <= 8:
            continue
        global_points = points+np.array([ox,oy],np.float32)
        bounds = (*global_points.min(axis=0),*global_points.max(axis=0))
        agreement = overlap(bounds,box)
        if agreement >= .6 and np.min(points) > 1 and np.all(points[:,0] < crop.shape[1]-2) and np.all(points[:,1] < crop.shape[0]-2):
            choices.append((agreement,global_points,round(width),round(height)))
    if not choices:
        return None,None
    _,points,w,h = max(choices,key=lambda item:item[0])
    dst = np.array([(0,0),(w-1,0),(w-1,h-1),(0,h-1)],np.float32)
    transform = cv2.getPerspectiveTransform(points,dst)
    working = cv2.warpPerspective(image,transform,(w,h),flags=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(working,3,3,4,4,cv2.BORDER_REPLICATE),points.tolist()
