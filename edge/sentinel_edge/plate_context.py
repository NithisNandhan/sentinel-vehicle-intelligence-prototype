"""Native-pixel context searches and bounded same-frame fragment recovery.

Nearby boxes only trigger another detector call. They never become a joined
OCR string or an extra temporal observation. Heuristics are development values.
"""
from collections import Counter
import math


def expanded_box(box, shape, fraction=.20, minimum=48, maximum=256):
    x1,y1,x2,y2=box
    px=min(maximum,max(minimum,math.ceil((x2-x1)*fraction)))
    py=min(maximum,max(minimum,math.ceil((y2-y1)*fraction)))
    return [max(0,math.floor(x1-px)),max(0,math.floor(y1-py)),
            min(shape[1],math.ceil(x2+px)),min(shape[0],math.ceil(y2+py))]


def fragment_pair(a,b):
    """Horizontal neighbours with substantial vertical overlap, not nested boxes."""
    if a[0]>b[0]:a,b=b,a
    aw,ah=a[2]-a[0],a[3]-a[1];bw,bh=b[2]-b[0],b[3]-b[1]
    if min(aw,ah,bw,bh)<=0:return False
    return (b[0]>=a[0]+.35*aw and b[2]>=a[2]+.20*bw
            and -.25*min(aw,bw)<=b[0]-a[2]<=.15*min(aw,bw)
            and min(a[3],b[3])-max(a[1],b[1])>=.65*min(ah,bh)
            and max(ah,bh)/min(ah,bh)<=3.)


def spans_fragments(candidate,a,b):
    """Require detector evidence through both fragments, not a larger empty ROI."""
    if not fragment_pair(a,b):return False
    centres=[((v[0]+v[2])/2,(v[1]+v[3])/2) for v in (a,b)]
    return (all(candidate[0]<=x<=candidate[2] and candidate[1]<=y<=candidate[3] for x,y in centres)
            and candidate[2]-candidate[0]>=.85*(max(a[2],b[2])-min(a[0],b[0])))


def context_proposals(image,vehicles,search,*,max_recoveries=2):
    metrics=Counter();proposals=[]
    def detect(context,kind):
        x,y,x2,y2=context;region=image[y:y2,x:x2]
        if not region.size:return None
        local=search.box(region);metrics[kind]+=1
        if local is None:return None
        a,b,c,d=[round(v) for v in local]
        a,b,c,d=max(0,a),max(0,b),min(x2-x,c),min(y2-y,d)
        if c-a<24 or d-b<8:return None
        quad=getattr(search,'last_text_quad',None)
        return dict(box=[a+x,b+y,c+x,d+y],confidence=float(search.last_confidence or 0.),
                    quad=[(qx-a,qy-b) for qx,qy in quad] if quad is not None else None,
                    crop_origin=kind,search_box=list(context),completion=getattr(search,'last_completion',None))
    for vehicle in vehicles:
        box=expanded_box(vehicle['box'],image.shape)
        proposal=detect(box,'padded_search')
        if proposal:proposals.append(proposal)
    # At most two extra calls per sampled frame, regardless of crowded scenes.
    originals=list(proposals)
    pairs=[(a,b) for i,a in enumerate(originals) for b in originals[i+1:]
           if fragment_pair(a['box'],b['box'])]
    for a,b in pairs[:max_recoveries]:
        union=[min(a['box'][0],b['box'][0]),min(a['box'][1],b['box'][1]),
               max(a['box'][2],b['box'][2]),max(a['box'][3],b['box'][3])]
        recovered=detect(expanded_box(union,image.shape,.20,24,128),'fragment_redetection')
        if recovered and spans_fragments(recovered['box'],a['box'],b['box']):
            recovered['recovered_from']=[a['box'],b['box']]
            proposals=[p for p in proposals if p is not a and p is not b]
            proposals.append(recovered);metrics['fragments_recovered']+=1
        else:metrics['fragment_recovery_rejected']+=1
    return proposals,metrics
