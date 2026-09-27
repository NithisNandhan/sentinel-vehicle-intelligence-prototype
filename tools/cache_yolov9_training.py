"""Pin the original YOLOv9 training implementation in the experiment cache."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

root = Path('/models/experiments/yolov9-training')
root.mkdir(parents=True,exist_ok=True)
if (root/'source-receipt.json').exists():
    # Reusing an experiment must not overwrite its pinned training source.
    receipt = json.loads((root/'source-receipt.json').read_text())
    if not all((root/name).is_file() for name in ('train.py','models/yolo.py','utils/general.py')):
        raise ValueError('Incomplete cached source; preserve and investigate')
    print(json.dumps({'revision':receipt['revision'],'status':'existing_pin_preserved'}),flush=True)
    raise SystemExit(0)
with urllib.request.urlopen('https://api.github.com/repos/WongKinYiu/yolov9/commits/main',timeout=30) as response:
    revision = json.load(response)['sha']
url = 'https://api.github.com/repos/WongKinYiu/yolov9/zipball/'+revision
with urllib.request.urlopen(url,timeout=60) as response:
    data = response.read(32*1024**2+1)
if len(data)>32*1024**2:
    raise ValueError('Source archive exceeds bound')
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for item in archive.infolist():
        relative = Path(*Path(item.filename).parts[1:])
        target = (root/relative).resolve()
        if not target.is_relative_to(root.resolve()) or item.file_size>8*1024**2:
            raise ValueError('Unsafe source archive')
        if not relative.parts or item.is_dir():
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.read(item))
(root/'source-receipt.json').write_text(json.dumps({'revision':revision,'url':url,
    'sha256':hashlib.sha256(data).hexdigest(),'license':'GPL-3.0','production_dependency':False},indent=2))
for name in ('requirements.txt','train.py','models/yolo.py','utils/loggers/__init__.py'):
    print('SOURCE',name,flush=True)
    print((root/name).read_text()[:15000],flush=True)
