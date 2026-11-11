"""Opt-in native-pixel crop completion, independent of OCR text and labels.

Keep the original detector as the seed. Related enclosing text and closed plate
boundaries can extend it; unrelated nearby text cannot. At most two focused
retries per sampled frame use the existing models, never a new OCR vote.
"""
from collections import Counter
import math

from .plate_proposals import checked_quad,with_margin


def area(box):
    return max(0,box[2]-box[0])*max(0,box[3]-box[1])


def intersects_another_plate(seed,candidate,raw):
    """Do not absorb a separate, non-overlapping positive plate proposal."""
    for box,source,*_ in raw:
        if source not in ('plate','tile') or not area(box):continue
        intersection=max(0,min(seed[2],box[2])-max(seed[0],box[0]))*max(0,min(seed[3],box[3])-max(seed[1],box[1]))
        if intersection/min(area(seed),area(box))>=.2:continue
        cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        if candidate[0]<=cx<=candidate[2] and candidate[1]<=cy<=candidate[3]:return True
    return False


def extends_same_plate(seed,candidate):
    """Strong containment and compatible height, not arbitrary box union."""
    if not area(seed) or not area(candidate):return False
    sw,sh=seed[2]-seed[0],seed[3]-seed[1]
    cw,ch=candidate[2]-candidate[0],candidate[3]-candidate[1]
    intersection=max(0,min(seed[2],candidate[2])-max(seed[0],candidate[0]))*max(0,min(seed[3],candidate[3])-max(seed[1],candidate[1]))
    # Detector padding is not character evidence; text outlines need to cover
    # the central seed content, not all surrounding background pixels.
    core=(seed[0]+.12*sw,seed[1]+.12*sh,seed[2]-.12*sw,seed[3]-.12*sh)
    core_overlap=max(0,min(core[2],candidate[2])-max(core[0],candidate[0]))*max(0,min(core[3],candidate[3])-max(core[1],candidate[1]))
    return (intersection/area(seed)>=.45 and core_overlap/area(core)>=.75 and 1.15<=area(candidate)/area(seed)<=6
            and .65<=ch/sh<=1.8 and 1.2<=cw/ch<=8
            and cw>=1.15*sw and abs((candidate[1]+candidate[3])-(seed[1]+seed[3]))/2<=.45*sh)


def stroke_boxes(image):
    import cv2
    import numpy as np
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);h,w=gray.shape
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    alternatives=[]
    for binary in (mask,255-mask):
        _,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
        boxes=[(int(x),int(y),int(x+cw),int(y+ch)) for x,y,cw,ch,a in stats[1:]
               if .22*h<=ch<=.93*h and .06<=cw/ch<=1.2 and a>=5 and .1<=a/(cw*ch)<=.95]
        alternatives.append(boxes)
    return max(alternatives,key=len)


def completeness(image):
    """Selection-only heuristic; no plate text or confidence calibration."""
    boxes=stroke_boxes(image);h,w=image.shape[:2]
    if not boxes:return .25
    edge=sum(x1<.035*w or x2>.965*w or y1<.025*h or y2>.975*h for x1,y1,x2,y2 in boxes)
    # Sparse, tall glyphs in a short box are a possible single-row fragment.
    single_row=sorted(b[3]-b[1] for b in boxes)[len(boxes)//2]>.45*h
    fragment=len(boxes)<6 and single_row and w/h<3
    return max(.1,min(1.,(.45 if fragment else .85)-.12*edge))


def boundary_extensions(image,seed):
    """Closed quadrilateral candidates containing the seed in the original ROI."""
    import cv2
    import numpy as np
    x1,y1,x2,y2=seed;w,h=x2-x1,y2-y1
    left,top=max(0,int(x1-2*w)),max(0,int(y1-.8*h))
    right,bottom=min(image.shape[1],int(x2+2*w)),min(image.shape[0],int(y2+.8*h))
    region=image[top:bottom,left:right]
    if not region.size:return []
    gray=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),35,110)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    proposals=[]
    for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:80]:
        perimeter=cv2.arcLength(contour,True);poly=cv2.approxPolyDP(contour,.025*perimeter,True)
        if len(poly)!=4 or not cv2.isContourConvex(poly):continue
        points=poly.reshape(4,2).astype(float);points[:,0]+=left;points[:,1]+=top
        centre=points.mean(axis=0);points=points[np.argsort(np.arctan2(points[:,1]-centre[1],points[:,0]-centre[0]))]
        # Angle order starts at the upper-left for ordinary, non-upside-down plates.
        points=np.roll(points,-int(np.argmin(points.sum(axis=1))),axis=0)
        quad=checked_quad(points.tolist(),image.shape)
        if quad is None:continue
        box=(min(p[0] for p in quad),min(p[1] for p in quad),max(p[0] for p in quad),max(p[1] for p in quad))
        if not extends_same_plate(seed,box):continue
        if cv2.contourArea(points.astype(np.float32))/area(box)<.65:continue
        crop=image[math.floor(box[1]):math.ceil(box[3]),math.floor(box[0]):math.ceil(box[2])]
        if len(stroke_boxes(crop))<3:continue
        proposals.append((box,quad))
    return proposals


class CompletePlateSearch:
    """ProposalPool-compatible adapter; legacy behaviour remains the default."""
    def __init__(self,primary,*,max_retries=2):
        self.primary=primary;self.max_retries=max_retries;self.remaining=0
        self.metrics=Counter();self.last_text_quad=None;self.last_confidence=None
        self.last_calls=0;self.last_fallback=False;self.last_completion=None

    @property
    def detector(self):return self.primary.detector

    def begin_frame(self):
        self.primary.begin_frame();self.remaining=self.max_retries

    def diagnostics(self):
        return dict(**self.primary.diagnostics(),crop_completion=dict(self.metrics),
                    text_detector_device=getattr(self.primary.text_detector,'device',None))

    def box(self,image):
        seed=self.primary.box(image)
        self.last_text_quad=self.primary.last_text_quad;self.last_confidence=self.primary.last_confidence
        self.last_calls=self.primary.last_calls;self.last_fallback=self.primary.last_fallback
        self.last_completion=None
        if seed is None:return None
        original=tuple(seed);candidates=[]
        # A complete text outline can enclose a partial detector box even when
        # IoU is low. Positive plate support still comes from the original seed.
        raw=getattr(self.primary,'last_raw_proposals',())
        for item in raw:
            box,source=item[:2]
            if source=='text' and extends_same_plate(seed,box):
                candidates.append((tuple(box),item[3] if len(item)>3 else None,'enclosing_text'))
        x1,y1,x2,y2=[int(v) for v in seed]
        native=image[y1:y2,x1:x2]
        if not native.size:return seed
        score=completeness(native)
        if score<.7:
            candidates.extend((box,quad,'closed_boundary') for box,quad in boundary_extensions(image,seed))
            if not candidates and self.remaining and self.primary.text_detector is not None:
                self.remaining-=1;self.metrics['focused_retries']+=1
                w,h=x2-x1,y2-y1
                left,top=max(0,x1-2*w),max(0,y1-h)
                right,bottom=min(image.shape[1],x2+2*w),min(image.shape[0],y2+h)
                region=image[top:bottom,left:right]
                self.last_calls+=1
                for polygon in self.primary.text_detector.polygons(region):
                    quad=checked_quad(polygon,region.shape)
                    if quad is None:continue
                    quad=tuple((x+left,y+top) for x,y in quad)
                    box=(min(p[0] for p in quad),min(p[1] for p in quad),max(p[0] for p in quad),max(p[1] for p in quad))
                    if extends_same_plate(seed,box):candidates.append((box,quad,'focused_text'))
        best=None
        for box,quad,source in candidates[:12]:
            if intersects_another_plate(original,box,raw):
                self.metrics['other_plate_veto']+=1;continue
            # Refinement may add context, never remove original plate pixels.
            union=(min(box[0],original[0]),min(box[1],original[1]),max(box[2],original[2]),max(box[3],original[3]))
            output=with_margin(tuple(round(v) for v in union),image.shape)
            crop=image[output[1]:output[3],output[0]:output[2]]
            newscore=completeness(crop)
            if newscore<score:continue
            rank=(newscore,min(3.,area(box)/area(seed)))
            if best is None or rank>best[0]:best=(rank,output,quad,source)
        if best:
            _,seed,quad,source=best
            # A text outline can cover just one line even when its enclosing
            # box is wider. Do not rectify away existing seed glyphs.
            if quad:
                import cv2
                import numpy as np
                polygon=np.array(quad,np.float32)
                for a,b,c,d in stroke_boxes(native):
                    points=((a+x1,b+y1),(c+x1,b+y1),(c+x1,d+y1),(a+x1,d+y1))
                    if any(cv2.pointPolygonTest(polygon,point,True)<-2 for point in points):quad=None;break
            self.last_text_quad=tuple(quad) if quad else None
            score=max(score,best[0][0]);self.metrics[source]+=1
            self.last_fallback=True
        self.last_completion=dict(seed_box=list(original),box=list(seed),selection_score=score,
                                  recovered=bool(best),method=best[3] if best else 'original')
        return seed
