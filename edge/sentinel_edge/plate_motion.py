"""Forward/backward checked optical flow supporting plate identity only.

Tracking uses a smaller grayscale image; saved/OCR crops always use the original
frame. Flow cannot create OCR observations, override text conflicts or assign a
vehicle owner. Failed/ambiguous flow simply leaves the appearance gate in force.
"""
import math
from collections import Counter


class PlateMotion:
    def __init__(self,*,width=1280,max_active=64):
        self.width=width;self.max_active=max_active;self.states={};self.previous=None
        self.pts=None;self.scale=None;self.metrics=Counter()

    def advance(self,image,pts_ms):
        import cv2
        import numpy as np
        if self.pts is not None and pts_ms<=self.pts:raise ValueError('Forward video PTS required')
        scale=min(1.,self.width/image.shape[1])
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if scale!=1:gray=cv2.resize(gray,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
        if self.previous is not None and (self.previous.shape!=gray.shape or pts_ms-self.pts>1000):
            self.states.clear();self.metrics['scene_reset']+=1
        self.scale=scale
        active=[(key,s) for key,s in self.states.items() if pts_ms-s['observed']<=1000]
        self.states=dict(active)
        if self.previous is not None and active:
            sizes=[len(s['points']) for _,s in active];points=np.concatenate([s['points'] for _,s in active]).astype(np.float32)
            forward,ok,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None,winSize=(21,21),maxLevel=3)
            if forward is None:self.states.clear();self.metrics['lost_batches']+=1
            else:
                backward,back_ok,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,forward,None,winSize=(21,21),maxLevel=3)
                offset=0
                for (key,state),size in zip(active,sizes):
                    old=points[offset:offset+size].reshape(-1,2);new=forward[offset:offset+size].reshape(-1,2)
                    good=np.zeros(size,dtype=bool) if backward is None else (
                        ok[offset:offset+size].ravel().astype(bool)&back_ok[offset:offset+size].ravel().astype(bool)&
                        (np.linalg.norm(backward[offset:offset+size].reshape(-1,2)-old,axis=1)<=1.))
                    offset+=size
                    good&=np.isfinite(new).all(axis=1)
                    good&=(new[:,0]>=0)&(new[:,0]<gray.shape[1])&(new[:,1]>=0)&(new[:,1]<gray.shape[0])
                    if good.sum()<8 or good.mean()<.65:
                        self.states.pop(key,None);self.metrics['lost_support']+=1;continue
                    transform,inliers=cv2.estimateAffinePartial2D(old[good],new[good],method=cv2.RANSAC,ransacReprojThreshold=1.5)
                    if transform is None or inliers.mean()<.75:
                        self.states.pop(key,None);self.metrics['lost_geometry']+=1;continue
                    a,b=transform[0,0],transform[1,0];factor=math.hypot(a,b)
                    if not .8<=factor<=1.25:
                        self.states.pop(key,None);self.metrics['scale_rejected']+=1;continue
                    corners=state['corners']*scale
                    moved=corners@transform[:,:2].T+transform[:,2]
                    state['corners']=moved/scale
                    state['box']=[float(moved[:,0].min()/scale),float(moved[:,1].min()/scale),
                                  float(moved[:,0].max()/scale),float(moved[:,1].max()/scale)]
                    state['points']=new[good][inliers.ravel().astype(bool)].reshape(-1,1,2)
                    state['steps']+=1;self.metrics['supported_updates']+=1
        self.previous=gray;self.pts=pts_ms

    def observe(self,identity,box):
        import cv2
        import numpy as np
        # A newly accepted detection supersedes every earlier prediction,
        # including when its crop cannot provide enough fresh tracking points.
        self.states.pop(identity,None)
        # Re-seed only within the detected plate, never from the whole vehicle.
        x1,y1,x2,y2=[round(v*self.scale) for v in box]
        x1,y1=max(0,x1),max(0,y1);x2,y2=min(self.previous.shape[1],x2),min(self.previous.shape[0],y2)
        if x2<=x1 or y2<=y1:return
        roi=self.previous[y1:y2,x1:x2]
        points=cv2.goodFeaturesToTrack(roi,maxCorners=40,qualityLevel=.02,minDistance=3,blockSize=3)
        if points is None or len(points)<8:
            self.metrics['reseed_without_features']+=1;return
        span=np.ptp(points.reshape(-1,2),axis=0)
        if span[0]<.3*(x2-x1) or span[1]<.3*(y2-y1):return
        points+=np.array([x1,y1],dtype=np.float32)
        self.states[identity]=dict(box=list(box),corners=np.array([[box[0],box[1]],[box[2],box[1]],
                                  [box[2],box[3]],[box[0],box[3]]],dtype=float),
                                  points=points,observed=self.pts,steps=0)
        while len(self.states)>self.max_active:
            del self.states[min(self.states,key=lambda k:self.states[k]['observed'])]
            self.metrics['capacity_evictions']+=1

    def support(self,box):
        from .owned_plate import box_iou
        # A prediction needs at least two decoded-frame motion checks.
        result={}
        for identity,state in self.states.items():
            if state['steps']<2:continue
            overlap=box_iou(box,state['box'])
            if overlap>=.30:result[identity]=overlap
        return result
