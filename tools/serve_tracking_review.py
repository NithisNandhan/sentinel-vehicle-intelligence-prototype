"""Local tracking replay and durable human review; no live inference or alerts."""
import argparse
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import urlsplit

STATUSES = {'pending','vehicle','non_vehicle','mixed','uncertain'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class ReviewStore:
    def __init__(self,root,video):
        self.root,self.video=Path(root).resolve(),Path(video).resolve()
        self.report=json.loads((self.root/'report.json').read_bytes())
        self.report_sha=digest(self.root/'report.json')
        if digest(self.video)!=self.report['source_sha256']:raise ValueError('Wrong source video')
        self.lock=threading.RLock()
        self.ids={r['passage_id'] for r in self.report['strategies']['screened']['records']}
        self.assets={p['path'] for r in self.report['strategies']['screened']['records'] for p in r['preview_paths']}
        self.review=dict(schema_version=1,report_sha256=self.report_sha,revision=0,labeller='',
                         labels={pid:dict(status='pending',note='') for pid in sorted(self.ids)})
        if (self.root/'vehicle-review.json').exists():
            value=json.loads((self.root/'vehicle-review.json').read_bytes())
            self.validate(value)
            self.review=value

    def validate(self,value):
        if not isinstance(value,dict) or value.get('schema_version')!=1 or value.get('report_sha256')!=self.report_sha:
            raise ValueError('Review belongs to another frozen run')
        if type(value.get('revision')) is not int or value['revision']<0:raise ValueError('Invalid revision')
        if not isinstance(value.get('labeller'),str) or len(value['labeller'])>160:raise ValueError('Invalid reviewer')
        labels=value.get('labels')
        if not isinstance(labels,dict) or set(labels)!=self.ids:raise ValueError('Wrong passage inventory')
        for label in labels.values():
            if (not isinstance(label,dict) or set(label)!={'status','note'} or label.get('status') not in STATUSES
                    or not isinstance(label.get('note'),str) or len(label['note'])>2000):
                raise ValueError('Invalid label')

    def save(self,value):
        with self.lock:
            self.validate(value)
            if value['revision']!=self.review['revision']:raise RuntimeError('Review changed in another tab; download your draft and reload')
            result={**value,'revision':value['revision']+1}
            tmp=self.root/'vehicle-review.tmp'
            tmp.write_bytes((json.dumps(result,indent=2)+'\n').encode())
            tmp.replace(self.root/'vehicle-review.json')
            self.review=result
            return result


def make_handler(store,port):
    allowed={f'127.0.0.1:{port}',f'localhost:{port}'}
    timeline=gzip.compress((store.root/'timeline.json').read_bytes())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def handle(self):
            try:super().handle()
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
        def safe(self):
            host=self.headers.get('Host','')
            return host in allowed and self.headers.get('Origin','http://'+host)=='http://'+host
        def reply(self,status,body,kind='application/json',headers=None):
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            for k,v in (headers or {}).items():self.send_header(k,v)
            self.end_headers()
            if self.command!='HEAD':self.wfile.write(body)
        def document(self,status,value):self.reply(status,json.dumps(value,allow_nan=False).encode())
        def video(self):
            size=store.video.stat().st_size;start,end,status=0,size-1,200
            value=self.headers.get('Range')
            if value:
                match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
                if not match or not any(match.groups()):self.reply(416,b'',headers={'Content-Range':f'bytes */{size}'});return
                first,last=match.groups()
                if first:start=int(first);end=min(int(last),size-1) if last else size-1
                else:start=max(0,size-int(last))
                if start>=size or start>end:self.reply(416,b'',headers={'Content-Range':f'bytes */{size}'});return
                status=206
            self.send_response(status);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
            self.send_header('Content-Length',str(end-start+1));self.send_header('Cache-Control','no-store')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command=='HEAD':return
            with store.video.open('rb') as source:
                source.seek(start);remaining=end-start+1
                while remaining:
                    block=source.read(min(1024*1024,remaining))
                    if not block:break
                    self.wfile.write(block);remaining-=len(block)
        def do_HEAD(self):self.do_GET()
        def do_GET(self):
            if not self.safe():self.document(403,{'error':'Local review only'});return
            path=urlsplit(self.path).path
            if path=='/':self.reply(200,Path(__file__).with_name('tracking_review.html').read_bytes(),'text/html; charset=utf-8')
            elif path=='/video':self.video()
            elif path=='/api/report':self.document(200,store.report)
            elif path=='/api/timeline':self.reply(200,timeline,headers={'Content-Encoding':'gzip'})
            elif path=='/api/review':
                with store.lock:self.document(200,store.review)
            elif path in store.assets or path.lstrip('/') in store.assets:
                relative=path.lstrip('/')
                if re.fullmatch(r'assets/V\d+-\d+\.jpg',relative):self.reply(200,(store.root/relative).read_bytes(),'image/jpeg')
                else:self.document(404,{'error':'Unknown image'})
            else:self.document(404,{'error':'Not found'})
        def do_POST(self):
            if not self.safe():self.document(403,{'error':'Local review only'});return
            if self.path!='/api/review':self.document(404,{'error':'Not found'});return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.document(415,{'error':'JSON required'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2*1024*1024:raise ValueError('Invalid review size')
                self.document(200,store.save(json.loads(self.rfile.read(length))))
            except RuntimeError as e:self.document(409,{'error':str(e)})
            except (ValueError,TypeError) as e:self.document(400,{'error':str(e)})
            except OSError:self.document(507,{'error':'Local save failed; download your draft'})
    return Handler


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--video',type=Path,default=Path('data/recordings/footage.mp4'));p.add_argument('--port',type=int,default=7798)
    args=p.parse_args();store=ReviewStore(args.run,args.video)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),make_handler(store,args.port));server.daemon_threads=True
    print(f'Tracking review: http://127.0.0.1:{args.port}/',flush=True);server.serve_forever()
