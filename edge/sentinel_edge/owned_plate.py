"""Conservative plate ownership for supplied vehicle observations, not identity proof."""
from collections import Counter
import math

from .plate_continuity import continuity_reason


def box_iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def valid_box(box):
    return (isinstance(box,(list,tuple)) and len(box)==4
            and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in box)
            and box[2]>box[0] and box[3]>box[1])


class OwnedPlateRouter:
    """At most one nonambiguous plate observation per qualified vehicle per PTS.

    All visible tracks participate in overlap checks, including tentative and
    excluded tracks. A neighbour cannot silently supply another vehicle's crop.
    """
    def __init__(self,excluded=()):
        self.excluded=set(excluded);self.counters=Counter();self.previous={};self.generations=Counter()
        self.last_pts=None

    def route(self,proposals,vehicles,pts_ms):
        if type(pts_ms) is not int or pts_ms<0 or (self.last_pts is not None and pts_ms<=self.last_pts):
            raise ValueError('Forward integer video PTS required')
        self.last_pts=pts_ms
        ids=[v['id'] for v in vehicles]
        if len(set(ids))!=len(ids):raise ValueError('Duplicate vehicle identity in one frame')
        ordered=sorted((p for p in proposals if valid_box(p.get('box'))),
                       key=lambda p:-p.get('confidence',0.))
        unique=[]
        for proposal in ordered:
            if any(box_iou(proposal['box'],p['box'])>=.5 for p in unique):
                self.counters['duplicate_proposals']+=1;continue
            unique.append(proposal)
        owners={}
        for proposal in unique:
            b=proposal['box'];area=(b[2]-b[0])*(b[3]-b[1]);matching=[]
            for vehicle in vehicles:
                v=vehicle['box']
                if not valid_box(v):continue
                overlap=max(0,min(v[2],b[2])-max(v[0],b[0]))*max(0,min(v[3],b[3])-max(v[1],b[1]))
                if overlap/area>=.9:matching.append(vehicle)
            if len(matching)!=1:
                self.counters['ambiguous_owner' if matching else 'unassigned_owner']+=1;continue
            owner=matching[0]
            if owner['id'] in self.excluded:
                self.counters['review_excluded']+=1;continue
            if not owner['qualified']:
                self.counters['tentative_owner']+=1;continue
            owners.setdefault(owner['id'],[]).append(proposal)
        accepted=[]
        for owner,values in owners.items():
            if len(values)!=1:
                self.counters['multiple_plates_same_owner']+=len(values);continue
            proposal=values[0];old=self.previous.get(owner)
            reason=continuity_reason(old[0] if old else None,old[1] if old else None,proposal['box'],pts_ms)
            if old is None or reason:
                self.generations[owner]+=1
                if reason:self.counters['continuity_'+reason]+=1
            group=f'{owner}-P{self.generations[owner]:03d}'
            self.previous[owner]=(proposal['box'],pts_ms)
            accepted.append({**proposal,'owner':owner,'group':group,'position_ms':pts_ms})
            self.counters['owned_observations']+=1
        return accepted
