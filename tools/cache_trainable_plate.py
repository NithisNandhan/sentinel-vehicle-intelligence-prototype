"""Cache the selected detector's publisher-provided training counterpart.

Inspect checkpoint globals without executing pickle. A hash is an integrity
receipt, not a publisher signature. Actual loading is isolated from secrets.
"""
import hashlib
import json
import pickletools
import shutil
import urllib.request
import zipfile
from pathlib import Path

root = Path('/models/experiments/detectors')
name = 'yolo-v9-t-512-license-plates.pt'
url = 'https://github.com/ankandrew/open-image-models/releases/download/assets/'+name
path = root/name
expected_sha256 = '8e5bf34083b0f3fea275b627dbf00d82a92122b04d7b3472af62f908a94602a8'
if not path.exists():
    temporary = path.with_suffix('.download')
    with urllib.request.urlopen(url,timeout=60) as response, temporary.open('wb') as target:
        shutil.copyfileobj(response,target)
    if temporary.stat().st_size != 8328876:
        raise ValueError('Publisher asset size changed')
    temporary.replace(path)
if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
    raise ValueError('Reviewed publisher checkpoint changed; do not execute it')
with zipfile.ZipFile(path) as archive:
    members = [n for n in archive.namelist() if n.endswith('/data.pkl')]
    if len(members) != 1:
        raise ValueError('Unexpected checkpoint container')
    globals_ = sorted({arg for op,arg,_ in pickletools.genops(archive.read(members[0])) if op.name == 'GLOBAL'})
record = {'name':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'source':url,'publisher_checksum_available':False,'pickle_globals':globals_}
(root/'trainable-plate.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
