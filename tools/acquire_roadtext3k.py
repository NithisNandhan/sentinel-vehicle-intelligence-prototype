"""Bounded, resumable acquisition from the official RoadText-3K publisher.

Only one Indian archive is fetched. Annotation byte ranges are an inspection
optimisation, not proof of geographic membership; the archive supplies that.
Source media and labels stay in ignored data/. No credentials are used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.request

BASE = 'http://datasets.cvc.uab.es/roadtext3k/'
MARKER = re.compile(rb'"(\d+)": \{"1": \{"labels":')


def head(url):
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
        return dict(bytes=int(response.headers['Content-Length']),etag=response.headers.get('ETag'),
                    last_modified=response.headers.get('Last-Modified'))


def read_range(url,start,end,metadata):
    for attempt in range(4):
        try:
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{metadata["bytes"]}':
                    raise ValueError('Publisher did not return the requested byte range')
                if metadata['etag'] and response.headers.get('ETag')!=metadata['etag']:
                    raise ValueError('Publisher file changed during acquisition')
                result=response.read(end-start+2)
            if len(result)!=end-start+1:
                raise ValueError('Incomplete publisher byte range')
            return result
        except (OSError,ValueError):
            if attempt==3:raise
            time.sleep(min(2**attempt,8))


def locate_video(url,target,metadata):
    """Find an ordered root video key without downloading unrelated annotations."""
    low,high=0,metadata['bytes']
    iteration=0
    closest=None
    while low<high:
        iteration+=1
        if iteration>32:raise ValueError('Video-key search did not converge')
        mid=(low+high)//2
        block=read_range(url,mid,min(metadata['bytes']-1,mid+1024*1024-1),metadata)
        matches=list(MARKER.finditer(block))
        print(json.dumps(dict(stage='annotation_index',target=target,iteration=iteration,
            offset=mid,video_keys=[int(m[1]) for m in matches])),flush=True)
        if not matches:
            block=read_range(url,mid,min(metadata['bytes']-1,mid+8*1024*1024-1),metadata)
            matches=list(MARKER.finditer(block))
        if not matches:
            if high-low<=4*1024*1024:
                block=read_range(url,low,min(metadata['bytes']-1,high+4*1024*1024-1),metadata)
                eligible=[(int(m[1]),low+m.start()) for m in MARKER.finditer(block) if int(m[1])>=target]
                if eligible:return min(eligible)[1]
                if closest is not None:return closest[1]
                raise ValueError(f'Annotation video key {target} was not found')
            high=mid
            continue
        for match in matches:
            if int(match[1])==target:return mid+match.start()
            if int(match[1])>target and (closest is None or int(match[1])<closest[0]):
                closest=(int(match[1]),mid+match.start())
        first,last=int(matches[0][1]),int(matches[-1][1])
        if target<first:high=mid
        elif target>last:low=mid+matches[-1].start()+1
        else:return min((int(m[1]),mid+m.start()) for m in matches if int(m[1])>=target)[1]
        if high-low<=1 and closest is not None:return closest[1]
    if closest is not None:return closest[1]
    raise ValueError('Video key search failed')


def acquire(args):
    if not re.fullmatch(r'I_\d+\.tar\.gz',args.archive):
        raise ValueError('Only an explicit Indian archive is admitted')
    args.output.mkdir(parents=True,exist_ok=True)
    annotation_url=BASE+'roadtext-annotation-fixed.json'
    annotation_meta=head(annotation_url)
    annotation_path=args.output/f'annotations-{args.first_video}-{args.next_video-1}.json'
    if not annotation_path.exists():
        start=locate_video(annotation_url,args.first_video,annotation_meta)
        end=locate_video(annotation_url,args.next_video,annotation_meta)
        if not 0<end-start<300*1024*1024:raise ValueError('Annotation subset exceeds limit')
        print(json.dumps(dict(stage='annotations',bytes=end-start,first=args.first_video,next=args.next_video)),flush=True)
        chunks=[]
        step=8*1024*1024
        ranges=[(offset,min(end-1,offset+step-1)) for offset in range(start,end,step)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            chunks=list(pool.map(lambda bounds:read_range(annotation_url,*bounds,annotation_meta),ranges))
        content=b'{'+b''.join(chunks).rstrip(b', \r\n')+b'}'
        document=json.loads(content)
        if not document or not set(document).issubset(set(map(str,range(args.first_video,args.next_video)))):
            raise ValueError('Annotation range contains unexpected video identities')
        annotation_path.write_bytes(content)
        receipt=dict(url=annotation_url,metadata=annotation_meta,start=start,end_exclusive=end,
                     local_sha256=hashlib.sha256(content).hexdigest(),videos=len(document))
        annotation_path.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    document=json.loads(annotation_path.read_text())
    texts=set(); labelled_videos=0; plate_instances=0
    for frames in document.values():
        found=set()
        for frame in frames.values():
            for label in frame.get('labels') or []:
                if (any(isinstance(v,list) and len(v)>1 and v[1]=='Licence_Plates' for v in label.get('attributes',{}).values())
                        and isinstance(label.get('ocr'),str) and label['ocr'].strip()):
                    found.add(label['ocr']);plate_instances+=1
        texts.update(found);labelled_videos+=bool(found)
    print(json.dumps(dict(stage='annotation_admission',videos_with_plate_text=labelled_videos,
        distinct_source_texts=len(texts),plate_instances=plate_instances)),flush=True)
    if labelled_videos<10 or len(texts)<20:raise ValueError('Too few public plate transcriptions in this annotation subset')
    url=BASE+args.archive
    metadata=head(url)
    if metadata['bytes']>3*1024**3:raise ValueError('Archive exceeds the single-archive bound')
    destination=args.output/args.archive
    receipt_path=destination.with_suffix(destination.suffix+'.receipt.json')
    if destination.exists():
        if not receipt_path.exists():raise ValueError('Existing archive lacks a verified acquisition receipt')
        receipt=json.loads(receipt_path.read_text())
        if receipt['metadata']!=metadata:raise ValueError('Cached archive source identity changed')
        print(json.dumps(dict(stage='archive_cached',bytes=destination.stat().st_size)),flush=True)
        return
    parts=args.output/(args.archive+'.parts')
    parts.mkdir(exist_ok=True)
    identity=parts/'source.json'
    expected=dict(url=url,metadata=metadata)
    if identity.exists() and json.loads(identity.read_text())!=expected:
        raise ValueError('Partial download belongs to a different publisher file')
    identity.write_text(json.dumps(expected,indent=2))
    step=16*1024*1024
    bounds=[(offset,min(metadata['bytes']-1,offset+step-1)) for offset in range(0,metadata['bytes'],step)]
    def part(item):
        number,(start,end)=item
        path=parts/f'{number:04d}.part'
        if not path.exists() or path.stat().st_size!=end-start+1:
            temporary=path.with_suffix('.partial')
            temporary.write_bytes(read_range(url,start,end,metadata))
            temporary.replace(path)
        return number,path
    completed=0
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for number,path in pool.map(part,enumerate(bounds)):
            completed+=path.stat().st_size
            if number%8==0 or completed==metadata['bytes']:
                print(json.dumps(dict(stage='archive_download',completed_bytes=completed,total_bytes=metadata['bytes'],seconds=round(time.monotonic()-started,1))),flush=True)
    temporary=destination.with_suffix('.partial')
    digest=hashlib.sha256()
    with temporary.open('wb') as output:
        for number in range(len(bounds)):
            with (parts/f'{number:04d}.part').open('rb') as source:
                while blob:=source.read(1024*1024):output.write(blob);digest.update(blob)
    if temporary.stat().st_size!=metadata['bytes']:raise ValueError('Merged archive size mismatch')
    temporary.replace(destination)
    receipt_path.write_text(json.dumps(dict(url=url,metadata=metadata,sha256=digest.hexdigest(),
        source='Official CVC publisher, local research acquisition; redistribution terms not established'),indent=2))
    print(json.dumps(dict(stage='acquisition_complete',archive_bytes=metadata['bytes'],sha256=digest.hexdigest())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive',default='I_400.tar.gz')
    parser.add_argument('--first-video',type=int,default=2390)
    parser.add_argument('--next-video',type=int,default=2510)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,9))
    acquire(parser.parse_args())
