"""Bounded extra plate searches on new/reappearing vehicle parents."""


class PlateBurstSchedule:
    def __init__(self,*,interval_ms=100,searches=3,max_pending=128,max_per_frame=2):
        self.interval_ms=interval_ms;self.searches=searches;self.max_pending=max_pending
        self.max_per_frame=max_per_frame;self.pending={};self.previous={};self.cooldown={};self.pts=None

    def advance(self,parents,pts_ms,*,reset=False):
        if self.pts is not None and pts_ms<=self.pts:raise ValueError('Forward source timestamps required')
        self.pts=pts_ms
        if reset:self.pending.clear();self.previous.clear();self.cooldown.clear()
        active={p['id'] for p in parents}
        self.pending={k:v for k,v in self.pending.items() if k in active and pts_ms<=v['until']}
        self.cooldown={k:v for k,v in self.cooldown.items() if k in active}
        for parent in parents:
            key=parent['id']
            if (parent['observed'] and self.previous.get(key)!='visible'
                    and pts_ms-self.cooldown.get(key,-10000)>=1000 and len(self.pending)<self.max_pending):
                self.pending[key]=dict(next=pts_ms,until=pts_ms+500,left=self.searches)
                self.cooldown[key]=pts_ms
        self.previous={p['id']:p['status'] for p in parents}

    def due(self):
        available=[k for k,v in sorted(self.pending.items(),key=lambda pair:(pair[1]['next'],pair[0])) if v['next']<=self.pts]
        return set(available[:self.max_per_frame])

    def searched(self,identities):
        for key in identities:
            state=self.pending.get(key)
            if state is None or state['next']>self.pts:continue
            state['left']-=1
            if not state['left']:del self.pending[key]
            else:state['next']=self.pts+self.interval_ms
