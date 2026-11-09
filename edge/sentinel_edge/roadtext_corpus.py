"""RoadText plate-label admission, native boxes and grouped experimental splits."""
import hashlib
import math
import re
from pathlib import PurePosixPath

from .temporal import indian_plate_format


def archive_member_kind(name,prefix):
    """Admit numeric videos plus the publisher's one bundled annotation file."""
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or len(path.parts)!=2 or path.parts[0]!=prefix:
        raise ValueError('Unexpected or unsafe Indian archive member')
    if path.name==prefix.removeprefix('I_')+'.json':return 'annotations'
    if path.suffix=='.mp4' and path.stem.isdigit():return 'video'
    raise ValueError('Unexpected or unsafe Indian archive member')


def plate_text(label):
    attributes=label.get('attributes')
    if not isinstance(attributes,dict) or not any(
            isinstance(value,list) and len(value)>1 and value[1]=='Licence_Plates'
            for value in attributes.values()):
        return None,'not_plate_category'
    if label.get('category')=='Illegible' or attributes.get('Illegible') is True:
        return None,'illegible'
    raw=label.get('ocr')
    if not isinstance(raw,str) or not raw.strip():
        return None,'missing_transcription'
    if not re.fullmatch(r'[A-Za-z0-9 .:/\-]+',raw):
        return None,'unsupported_or_uncertain_text'
    text=re.sub(r'[ .:/\-]','',raw).upper()
    if not indian_plate_format(text):
        return None,'fragment_or_uncovered_format'
    return text,None


def native_box(value,width,height):
    try:
        numbers=[value[k] for k in ('x1','y1','x2','y2')]
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in numbers):
            return None
        x1,y1,x2,y2=numbers
        x1,y1=max(0,math.floor(x1)),max(0,math.floor(y1))
        x2,y2=min(width,math.ceil(x2)),min(height,math.ceil(y2))
        return (x1,y1,x2,y2) if x2-x1>=24 and y2-y1>=8 else None
    except (TypeError,KeyError):
        return None


def grouped_splits(rows,seed='roadtext3k-20261109'):
    """Connected video/registration components prevent both forms of leakage."""
    parents={}
    def find(key):
        parents.setdefault(key,key)
        if parents[key]!=key:parents[key]=find(parents[key])
        return parents[key]
    def join(a,b):
        a,b=find(a),find(b)
        parents[max(a,b)]=min(a,b)
    for row in rows:
        join('video:'+row['video_id'],'plate:'+row['transcription'])
    members={}
    for node in list(parents):members.setdefault(find(node),[]).append(node)
    decisions={}
    for representative,nodes in members.items():
        identity='|'.join(sorted(nodes))
        bucket=int(hashlib.sha256((seed+identity).encode()).hexdigest(),16)%10
        decisions[representative]='train' if bucket<5 else 'validation' if bucket<7 else 'test'
    return [dict(row,split=decisions[find('video:'+row['video_id'])],
                 split_group=hashlib.sha256('|'.join(sorted(members[find('video:'+row['video_id'])])).encode()).hexdigest()[:16]) for row in rows]


def check_splits(rows):
    identities={}; pixels={}
    for row in rows:
        if row['split'] not in ('train','validation','test'):raise ValueError('Invalid RoadText split')
        for key in ('video_id','transcription'):
            identity=(key,row[key])
            if identity in identities and identities[identity]!=row['split']:
                raise ValueError('RoadText video/registration split leakage')
            identities[identity]=row['split']
        digest=row['decoded_pixel_sha256']
        if digest in pixels and pixels[digest]!=row['split']:raise ValueError('RoadText pixel split leakage')
        pixels[digest]=row['split']
    return True


def blocked_splits(rows):
    """Keep source video blocks separate; quarantine plates spanning blocks.

    Connected components can join almost an entire road sequence. Partition
    ordered video IDs first, then discard every cross-block registration. No
    pixels or model predictions determine the partition or which side wins.
    """
    videos=sorted({r['video_id'] for r in rows},key=int)
    if len(videos)<5:raise ValueError('Too few videos for three source blocks')
    first=max(1,int(len(videos)*.5));second=max(first+1,int(len(videos)*.7))
    assignment={video:'train' if i<first else 'validation' if i<second else 'test'
                for i,video in enumerate(videos)}
    appearances={}
    for row in rows:appearances.setdefault(row['transcription'],set()).add(assignment[row['video_id']])
    crossing={text for text,splits in appearances.items() if len(splits)>1}
    selected=[dict(row,split=assignment[row['video_id']],split_group='video:'+row['video_id'])
              for row in rows if row['transcription'] not in crossing]
    return selected,dict(cross_block_registrations=len(crossing),
                         cross_block_crops=len(rows)-len(selected))
