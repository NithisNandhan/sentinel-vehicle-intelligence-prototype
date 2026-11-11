"""Native stroke/edge extent recovery for an incomplete single-row plate.

No OCR strings, labels or generated pixels. A connected row of character-sized
components may extend a detector seed. Other positive plate boxes veto growth.
"""
import math
from .crop_completion import extends_same_plate,intersects_another_plate
from .plate_evidence import character_support


def components(gray,low,high):
    import cv2
    _,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    options=[]
    for binary in (mask,255-mask):
        _,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
        boxes=[(int(x),int(y),int(x+w),int(y+h)) for x,y,w,h,a in stats[1:]
               if low<=h<=high and .07<=w/h<=1.4 and a>=6 and .1<=a/(w*h)<=.95]
        # Tiny enclosed character holes must not outvote full-sized glyphs.
        options.append(boxes)
    return max(options,key=lambda boxes:sum((b[3]-b[1])**2 for b in boxes))


def character_extent(image,seed,other_boxes=()):
    import cv2
    import numpy as np
    x,y,x2,y2=map(int,seed);w,h=x2-x,y2-y
    if min(w,h)<8 or not (0<=x<x2<=image.shape[1] and 0<=y<y2<=image.shape[0]):return None
    native=image[y:y2,x:x2];gray=cv2.cvtColor(native,cv2.COLOR_BGR2GRAY)
    glyphs=components(gray,.25*h,.93*h)
    if not 2<=len(glyphs)<=6:return None
    height=float(np.median([b[3]-b[1] for b in glyphs]))
    if height<.30*h or w/h>=3:return None
    points=np.array([((a+c)/2+x,(b+d)/2+y) for a,b,c,d in glyphs])
    slopes=[(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(points) for b in points[i+1:] if abs(b[0]-a[0])>.2*w]
    slope=float(np.clip(np.median(slopes) if slopes else 0,-.4,.4))
    intercept=float(np.median(points[:,1]-slope*points[:,0]))
    left,top=max(0,math.floor(x-2.5*w)),max(0,math.floor(y-h))
    right,bottom=min(image.shape[1],math.ceil(x2+2.5*w)),min(image.shape[0],math.ceil(y2+h))
    region=image[top:bottom,left:right]
    candidates=[]
    for a,b,c,d in components(cv2.cvtColor(region,cv2.COLOR_BGR2GRAY),.5*height,1.6*height):
        box=(a+left,b+top,c+left,d+top);cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
        if abs(cy-(slope*cx+intercept))<=.4*height:candidates.append(box)
    selected=[b for b in candidates if min(b[2],x2)>max(b[0],x)]
    if not selected:return None
    pending=[b for b in candidates if b not in selected]
    while pending:
        lo,hi=min(b[0] for b in selected),max(b[2] for b in selected)
        connected=[b for b in pending if min(abs(b[2]-lo),abs(b[0]-hi))<=.9*height]
        if not connected:break
        for box in connected:selected.append(box);pending.remove(box)
    new=[b for b in selected if b[2]<x+.1*w or b[0]>x2-.1*w]
    if len(new)<2:return None
    box=(min(x,min(b[0] for b in selected)-4),min(y,min(b[1] for b in selected)-4),
         max(x2,max(b[2] for b in selected)+4),max(y2,max(b[3] for b in selected)+4))
    box=(max(0,box[0]),max(0,box[1]),min(image.shape[1],box[2]),min(image.shape[0],box[3]))
    if not extends_same_plate(seed,box):return None
    if intersects_another_plate(seed,box,[(b,'plate',None) for b in other_boxes]):return None
    return dict(box=list(map(int,box)),method='character_row_extent',added_components=len(new),slope=slope)


def recover_readable_extent(image,seed,rows,tokens,other_boxes=()):
    """Do not grow reflection/window patterns without any original OCR support."""
    if character_support(rows,tokens)<1:return None
    return character_extent(image,seed,other_boxes)
