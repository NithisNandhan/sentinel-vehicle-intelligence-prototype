"""Loopback-only human review of a frozen full-video run; predictions stay hidden."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import io
import json
from pathlib import Path
import re
import struct
import sys
import threading
from urllib.parse import unquote,urlsplit
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'edge'))
from sentinel_edge.full_video_review import validate_review,score_review


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8'))
    temporary.replace(path)


class Conflict(ValueError):pass


class ReviewStore:
    def __init__(self,root,video):
        self.root=Path(root).resolve();self.video=Path(video).resolve();self.lock=threading.RLock()
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        self.sha=digest(self.root/'manifest.json')
        self.prediction_sha=digest(self.root/'predictions.json')
        self.predictions=json.loads((self.root/'predictions.json').read_text(encoding='utf-8'))
        if (self.predictions.get('manifest_sha256')!=self.sha
                or self.predictions.get('source_sha256')!=self.manifest['source']['sha256']):
            raise ValueError('Frozen run/prediction identity mismatch')
        if digest(self.video)!=self.manifest['source']['sha256']:raise ValueError('The review video changed')
        if not self.manifest.get('run',{}).get('reached_eof'):raise ValueError('The video run did not finish')
        for row in self.manifest['tracks']:
            for asset in [row.get('vehicle_preview'),*row.get('samples',[])]:
                if not asset:continue
                path=self.asset_path(asset['path'])
                if asset.get('image_sha256') and digest(path)!=asset['image_sha256']:
                    raise ValueError('A frozen review image changed')
        self.finalized=(self.root/'finalized.json').exists();self.report=None
        if self.finalized:
            frozen=json.loads((self.root/'finalized.json').read_text(encoding='utf-8'))
            if frozen['manifest_sha256']!=self.sha or frozen['predictions_sha256']!=self.prediction_sha:
                raise ValueError('Final report belongs to another frozen run')
            self.labels=frozen['labels'];self.report=frozen['report']
        elif (self.root/'labels.json').exists():
            self.labels=validate_review(self.manifest,json.loads((self.root/'labels.json').read_text(encoding='utf-8')),self.sha)
        else:
            self.labels=dict(schema_version=1,manifest_sha256=self.sha,labeller='',inventory_complete=False,
                revision=0,extras=[],labels=[dict(id=r['id'],status='pending',plate=None,duplicate_of=None,note='')
                    for r in self.manifest['tracks']])
            write_json(self.root/'labels.json',self.labels)

    def asset_path(self,relative):
        if not re.fullmatch(r'assets/[A-Za-z0-9_-][A-Za-z0-9_.-]{0,140}\.(?:png|jpe?g|webp)',relative) or '..' in relative:
            raise ValueError('Invalid evidence path')
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root/'assets'):raise ValueError('Evidence path escaped the run')
        return path

    def public(self):
        return dict(manifest=self.manifest,manifest_sha256=self.sha,labels=self.labels,finalized=self.finalized)

    def check_revision(self,payload):
        if self.finalized:raise Conflict('This review has been finalised and is read-only')
        if type(payload.get('revision')) is not int:raise ValueError('A numeric review revision is required')
        if payload.get('revision')!=self.labels['revision']:raise Conflict('The review changed in another tab; preserve your draft and reload')

    def save(self,payload,final=False):
        with self.lock:
            self.check_revision(payload)
            if payload.get('extras',[])!=self.labels['extras']:
                raise ValueError('Missed-vehicle evidence must be added through its capture control')
            clean=validate_review(self.manifest,payload,self.sha,require_complete=final)
            clean['revision']=self.labels['revision']+1
            if final:
                report=score_review(self.manifest,self.predictions,clean,self.sha)
                report['generated_at']=datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
                report['predictions_file_sha256']=self.prediction_sha
                report['review_asset_hashes']={extra['image_path']:digest(self.asset_path(extra['image_path'])) for extra in clean['extras']}
                write_json(self.root/'accuracy-report.json',report)
                write_json(self.root/'labels.final.json',clean)
                write_json(self.root/'finalized.json',dict(manifest_sha256=self.sha,predictions_sha256=self.prediction_sha,labels=clean,report=report))
                self.labels=clean;self.report=report;self.finalized=True
                return dict(report=report,revision=clean['revision'])
            previous=self.root/'labels.json'
            if previous.exists():(self.root/'labels.previous.json').write_bytes(previous.read_bytes())
            write_json(previous,clean);self.labels=clean
            return dict(revision=clean['revision'])

    def add_extra(self,payload):
        from PIL import Image
        with self.lock:
            self.check_revision(payload)
            box=payload.get('box');position=payload.get('position_ms')
            if not isinstance(box,list) or len(box)!=4 or any(isinstance(n,bool) or not isinstance(n,int) for n in box):
                raise ValueError('Capture box must use four native integer coordinates')
            uid='manual-'+uuid.uuid4().hex
            extra=dict(id=uid,position_ms=position,box=box,image_path=f'assets/{uid}.png')
            candidate={**self.labels,'extras':[*self.labels['extras'],extra],
                'labels':[*self.labels['labels'],dict(id=uid,status='pending',plate=None,duplicate_of=None,note='')]}
            clean=validate_review(self.manifest,candidate,self.sha)
            encoded=payload.get('image_png')
            if not isinstance(encoded,str) or len(encoded)>24*1024*1024:raise ValueError('Capture is too large')
            try:blob=base64.b64decode(encoded,validate=True)
            except ValueError:raise ValueError('Capture PNG encoding is invalid') from None
            if len(blob)<33 or blob[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Capture must be a native PNG')
            dimensions=struct.unpack('>II',blob[16:24])
            if dimensions!=(box[2]-box[0],box[3]-box[1]) or min(dimensions)<2:
                raise ValueError('Capture dimensions do not match the native selection')
            with Image.open(io.BytesIO(blob)) as image:
                if image.format!='PNG':raise ValueError('Capture must be PNG')
                image.verify()
            self.asset_path(extra['image_path']).write_bytes(blob)
            clean['revision']=self.labels['revision']+1
            write_json(self.root/'labels.json',clean);self.labels=clean
            return dict(extra=extra,revision=clean['revision'])


def make_handler(store,port):
    allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def handle(self):
            # Browsers cancel video ranges and idle keep-alive connections
            # during seeking. This is normal, not a failed analysis/session.
            try:super().handle()
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def log_message(self,*args):pass
        def safe(self):
            host=self.headers.get('Host','');origin=self.headers.get('Origin')
            return host in allowed and (not origin or origin=='http://'+host)
        def reply(self,status,body,kind='application/json',headers=None):
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            for key,value in (headers or {}).items():self.send_header(key,value)
            self.end_headers()
            if self.command!='HEAD':self.wfile.write(body)
        def json(self,status,value):self.reply(status,json.dumps(value,allow_nan=False).encode('utf-8'))
        def send_video(self):
            size=store.video.stat().st_size;start=0;end=size-1;status=200
            value=self.headers.get('Range')
            if value:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
                if not match or not any(match.groups()):
                    self.reply(416,b'',headers={'Content-Range':f'bytes */{size}'});return
                first,last=match.groups()
                if first:start=int(first);end=min(int(last),size-1) if last else size-1
                else:start=max(0,size-int(last))
                if start>=size or start>end:
                    self.reply(416,b'',headers={'Content-Range':f'bytes */{size}'});return
                status=206
            self.send_response(status);self.send_header('Content-Type','video/mp4')
            self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
            self.send_header('Cache-Control','no-store')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command=='HEAD':return
            try:
                with store.video.open('rb') as source:
                    source.seek(start);remaining=end-start+1
                    while remaining:
                        block=source.read(min(1024*1024,remaining))
                        if not block:break
                        self.wfile.write(block);remaining-=len(block)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_HEAD(self):self.do_GET()
        def do_GET(self):
            if not self.safe():self.json(403,dict(error='Local review access only'));return
            path=unquote(urlsplit(self.path).path)
            try:
                if path=='/':self.reply(200,Path(__file__).with_name('video_review.html').read_bytes(),'text/html; charset=utf-8')
                elif path=='/video':self.send_video()
                elif path=='/api/review':
                    with store.lock:self.json(200,store.public())
                elif path=='/api/export':
                    with store.lock:self.json(200,store.labels)
                elif path=='/api/report':
                    with store.lock:
                        if not store.finalized:self.json(404,dict(error='Finish human review before revealing predictions'))
                        else:self.json(200,store.report)
                elif path.startswith('/assets/'):
                    relative=path.lstrip('/')
                    with store.lock:
                        known={a['path'] for r in store.manifest['tracks'] for a in [r.get('vehicle_preview'),*r.get('samples',[])] if a}
                        known.update(e['image_path'] for e in store.labels['extras'])
                    if relative not in known:self.json(404,dict(error='Unknown review image'));return
                    self.reply(200,store.asset_path(relative).read_bytes(),'image/png')
                else:self.json(404,dict(error='Not found'))
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):return
            except (ValueError,OSError):self.json(400,dict(error='The requested review evidence is unavailable'))
        def do_POST(self):
            if not self.safe():self.json(403,dict(error='Local review access only'));return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':
                self.json(415,dict(error='JSON request required'));return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=25*1024*1024:raise ValueError('Request is empty or too large')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict):raise ValueError('Review document must be an object')
                path=urlsplit(self.path).path
                if path=='/api/labels':result=store.save(payload)
                elif path=='/api/finalize':result=store.save(payload,final=True)
                elif path=='/api/extras':result=store.add_extra(payload)
                else:self.json(404,dict(error='Not found'));return
                self.json(200,result)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):return
            except Conflict as error:self.json(409,dict(error=str(error)))
            except (ValueError,KeyError) as error:self.json(400,dict(error=str(error)))
            except OSError:self.json(507,dict(error='Could not save locally. Download your browser draft and check free disk space.'))
    return Handler


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--video',type=Path,default=Path('data/recordings/footage.mp4'))
    parser.add_argument('--port',type=int,default=7795)
    args=parser.parse_args()
    store=ReviewStore(args.run,args.video)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(store,args.port))
    server.daemon_threads=True
    print(f'Human plate review: http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()
