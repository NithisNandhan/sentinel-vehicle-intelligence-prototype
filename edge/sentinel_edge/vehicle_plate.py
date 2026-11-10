"""Vehicle ownership and multiple plate surfaces without duplicate frame votes."""
import math
from collections import Counter
from .owned_plate import box_iou
from .plate_identity import PlateIdentityTracker,_conflict


def deduplicate_views(proposals):
    """Suppress a nested plate fragment, preserving the larger native view.

Separate front/side views survive. A strong textual conflict preserves both
for diagnosis rather than deciding identity from containment alone.
"""
    unique=[];metrics=Counter()
    area=lambda p:(p['box'][2]-p['box'][0])*(p['box'][3]-p['box'][1])
    for current in sorted(proposals,key=lambda p:(-area(p),-p.get('confidence',0))):
        b=current['box'];duplicate=False
        for other in unique:
            a=other['box'];intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            nested=intersection/area(current)>=.90 and area(other)/area(current)<=6
            if ((box_iou(a,b)>=.5 or nested)
                    and not _conflict(other.get('reading',()),current.get('reading',()))):
                duplicate=True;metrics['nested_or_overlapping_plate_views']+=1;break
        if not duplicate:unique.append(current)
    return unique,metrics


class VehiclePlateRouter(PlateIdentityTracker):
    """Motion can resolve competing expanded vehicle boxes after plate continuity.

This does not infer ownership on predicted-only vehicle positions. Owner locking
still requires two distinct visible plate observations. Existing strong OCR
conflict and ambiguity gates remain active for plate association.
"""
    def __init__(self):
        super().__init__(retain_owned=True)
        self._vehicles=[]

    def route(self,proposals,vehicles,pts_ms):
        self._vehicles=list(vehicles)
        return super().route(proposals,self._vehicles,pts_ms)

    def _owner(self,proposal,vehicles,group=None,pts_ms=None):
        b=proposal['box'];area=(b[2]-b[0])*(b[3]-b[1])
        for vehicle in vehicles:
            if not vehicle.get('ownership_veto'):continue
            v=vehicle['box']
            overlap=max(0,min(b[2],v[2])-max(b[0],v[0]))*max(0,min(b[3],v[3])-max(b[1],v[1]))/area
            if overlap>=.20:return None,'reviewed_overlap_veto'
        ordinary=super()._owner(proposal,vehicles,group,pts_ms)
        if not group or not group.observations:return ordinary
        previous,then=group.observations[-1];dt=pts_ms-then
        if not 150<=dt<=1000:return ordinary
        b=proposal['box'];a=previous['box']
        velocity=[((b[i]+b[i+2])-(a[i]+a[i+2]))/(2*dt) for i in (0,1)]
        diagonal=max(40.,math.hypot(b[2]-b[0],b[3]-b[1]));candidates=[]
        for vehicle in vehicles:
            if (not vehicle.get('observed',True) or not vehicle['qualified']
                    or vehicle.get('observation_span_ms',0)<150):continue
            v=vehicle['box'];px=(v[2]-v[0])*.20;py=(v[3]-v[1])*.20
            extended=[v[0]-px,v[1]-py,v[2]+px,v[3]+py]
            coverage=max(0,min(b[2],extended[2])-max(b[0],extended[0]))*max(0,min(b[3],extended[3])-max(b[1],extended[1]))/((b[2]-b[0])*(b[3]-b[1]))
            if coverage<.90:continue
            motion=vehicle.get('velocity_px_per_ms')
            if motion is None:continue
            error=math.hypot(*(velocity[i]-motion[i] for i in (0,1)))*dt/diagonal
            candidates.append((error,vehicle['id']))
        candidates.sort()
        if candidates and candidates[0][0]<=.5 and (len(candidates)==1 or candidates[1][0]-candidates[0][0]>=.2):
            return candidates[0][1],'motion_unique_owner'
        return ordinary

    def _edge(self,group,current,pts_ms,identity):
        # A conflicting visible, uniquely established parent cannot silently
        # inherit another parent's OCR history, even if registration text agrees.
        hint,status=self._owner(current,self._vehicles,group,pts_ms)
        if group.owner_votes>=2 and hint is not None and group.candidate!=hint:
            self.counters['parent_conflict']+=1;return None
        return super()._edge(group,current,pts_ms,identity)


def one_view_per_timestamp(observations,tokens):
    """Select original evidence; side/front views in one frame supply one vote."""
    from .ctc_fusion import greedy_reading
    selected={};anchor=None;conflict=False;duplicates=0
    for observation in sorted(observations,key=lambda o:(o['position_ms'],o.get('source_id',''))):
        reading=greedy_reading(observation['rows'],tokens)
        if anchor is not None and _conflict(anchor,reading):conflict=True
        if anchor is None or sum(p>=.9 for _,p in reading)>sum(p>=.9 for _,p in anchor):anchor=reading
        pts=observation['position_ms']
        if pts in selected:duplicates+=1
        if pts not in selected or observation['quality']>selected[pts]['quality']:selected[pts]=observation
    return list(selected.values()),dict(same_frame_views_not_counted=duplicates,conflicting_plate_views=conflict)


def vehicle_signature(image,box):
    """Small tracking descriptor; never a vehicle identity assertion or OCR crop."""
    import cv2
    import numpy as np
    x1,y1,x2,y2=box
    region=image[max(0,round(y1)):min(image.shape[0],round(y2)),max(0,round(x1)):min(image.shape[1],round(x2))]
    if not region.size:return []
    small=cv2.resize(region,(32,16));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    gray-=gray.mean();norm=float(np.linalg.norm(gray))
    if norm:gray/=norm
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    colors=cv2.calcHist([hsv],[0,1],None,[16,4],[0,180,0,256]).ravel();colors/=max(float(np.linalg.norm(colors)),1e-6)
    return np.concatenate((gray*.8,colors*.2)).tolist()


class SceneBoundary:
    def __init__(self):self.gray=None;self.hist=None

    def changed(self,image):
        import cv2
        import numpy as np
        small=cv2.resize(image,(64,36));gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
        hist=cv2.calcHist([small],[0,1,2],None,[8,8,8],[0,256,0,256,0,256]);hist/=max(float(hist.sum()),1.)
        cut=self.gray is not None and float(np.abs(gray.astype(float)-self.gray).mean())>60 and cv2.compareHist(self.hist,hist,cv2.HISTCMP_BHATTACHARYYA)>.65
        self.gray=gray.astype(float);self.hist=hist
        return bool(cut)
