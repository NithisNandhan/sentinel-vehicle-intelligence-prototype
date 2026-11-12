"""Bounded native-view diagnostic using the unchanged plate detector.

Every view is searched even when full-ROI detection succeeds. No OCR, manual
geometry, labels, resizing or generated image detail is used here.
"""
from __future__ import annotations

import math
from .automatic_crops import overlap,windows

POLICY='full_view_protected_v2'


def _spread(views,shape,limit):
    """Farthest-centre sampling spreads a small budget across the whole ROI."""
    h,w=shape[:2]
    pool=list(views);result=[]
    def point(view):
        a,b,c,d=view
        return ((a+c)/(2*w),(b+d)/(2*h))
    while pool and len(result)<limit:
        if not result:
            index=0
        else:
            chosen=[point(v) for v in result]
            distances=[min((point(v)[0]-x)**2+(point(v)[1]-y)**2 for x,y in chosen) for v in pool]
            index=max(range(len(pool)),key=lambda i:distances[i])
        result.append(pool.pop(index))
    return result


def search_views(shape,max_calls=6):
    if isinstance(max_calls,bool) or not isinstance(max_calls,int) or not 1<=max_calls<=6:
        raise ValueError('Between one and six detector calls required')
    h,w=shape[:2]
    if min(h,w)<1:
        raise ValueError('Nonempty native image required')
    full=(0,0,w,h)
    candidates=[]
    if max(h,w)>640:
        size=640 if max(h,w)>=1280 else 512
        candidates=[(f'native-{size}',box) for box in _spread(windows(shape,size),shape,max_calls-1)]
    else:
        tw,th=max(1,round(w*.65)),max(1,round(h*.65))
        candidates=[('corner-tl',(0,0,tw,th)),('corner-br',(w-tw,h-th,w,h)),
                    ('corner-tr',(w-tw,0,w,th)),('corner-bl',(0,h-th,tw,h)),
                    ('lower-70',(0,round(h*.3),w,h))]
    result=[('full',full)];seen={full}
    for name,box in candidates:
        if box in seen:continue
        seen.add(box);result.append((name,box))
        if len(result)==max_calls:break
    return result[:max_calls]


def detect_views(image,scored_predict,max_calls=6):
    """Return ``(proposals, call_count)`` in unchanged source coordinates.

    ``scored_predict`` returns ``[(local_box, confidence), ...]``. Same-plate
    cross-view duplicates are full-view-first IoU suppressed, at most 32 kept.
    Extra views may add proposals but never replace a full-view proposal.
    The caller must not count alternate views as independent temporal evidence.
    """
    if getattr(image,'ndim',None)!=3 or image.shape[2]!=3 or not image.size:
        raise ValueError('Nonempty native colour image required')
    h,w=image.shape[:2];pending=[];calls=0
    for view,(x,y,x2,y2) in search_views(image.shape,max_calls):
        tile=image[y:y2,x:x2]
        predictions=scored_predict(tile);calls+=1
        for raw,confidence in predictions:
            if len(raw)!=4 or any(not math.isfinite(float(v)) for v in raw) or not math.isfinite(float(confidence)):
                continue
            a,b,c,d=map(float,raw)
            a,b,c,d=max(0.,a),max(0.,b),min(float(x2-x),c),min(float(y2-y),d)
            pw,ph=c-a,d-b
            if pw<=0 or ph<=0:
                continue
            if view!='full':
                if pw<24 or ph<8 or not 1.2<=pw/ph<=8:
                    continue
                cut=((a<=2 and x>0) or (b<=2 and y>0) or
                     (c>=x2-x-2 and x2<w) or (d>=y2-y-2 and y2<h))
                if cut:continue
            pending.append(dict(box=[a+x,b+y,c+x,d+y],confidence=float(confidence),view=view))
    result=[]
    for p in sorted(pending,key=lambda p:(p['view']!='full',-p['confidence'])):
        # The parent detector already performed its own suppression. Additional
        # cross-view policy must not remove one of its unchanged full outputs.
        if p['view']!='full' and any(overlap(p['box'],q['box'])>=.45 for q in result):continue
        result.append(p)
        if len(result)==32:break
    return result,calls
