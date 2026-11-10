"""Restore the exact public appearance encoders used by tracking diagnostics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

MODELS={
    'vehicle':dict(name='osnet_ain_x1_0_vehicle_reid.onnx',bytes=8836743,
        sha256='4aaad3e5db648618b0df3d2ff21c61323985ff9e50194c3d2edd4fb87c92d91f',
        url='https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx',
        license='https://raw.githubusercontent.com/sovrasov/deep-person-reid/vehicle_reid/LICENSE',
        source='https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/vehicle-reid-0001'),
    'ultralytics':dict(name='yolo26n-reid.onnx',bytes=9873245,
        sha256='8529c383197ae4c468eda535d1b165f8b4162cf17bf5fbcff49c7cb6455bc0bb',
        url='https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-reid.onnx',
        license='https://www.ultralytics.com/license',source='https://docs.ultralytics.com/modes/track/')}


def valid(path,model):
    return path.is_file() and path.stat().st_size==model['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==model['sha256']


def restore(output,key):
    model=MODELS[key];output.mkdir(parents=True,exist_ok=True);target=output/model['name']
    if not valid(target,model):
        temporary=target.with_suffix('.part')
        url=model['url']
        if 'github.com/' in url:url+='?sentinel_restore='+str(time.time_ns())
        request=urllib.request.Request(url,headers={'User-Agent':'Sentinel-local-benchmark'})
        with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as file:shutil.copyfileobj(response,file)
        if not valid(temporary,model):raise ValueError('Appearance asset failed pinned size/checksum verification')
        temporary.replace(target)
    (output/(model['name']+'.receipt.json')).write_text(json.dumps(model,indent=2)+'\n',encoding='utf-8')
    return dict(name=model['name'],bytes=model['bytes'],sha256=model['sha256'],verified=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('data/models/trackers'))
    parser.add_argument('--model',choices=tuple(MODELS),default='vehicle');args=parser.parse_args()
    print(json.dumps(restore(args.output,args.model)))
