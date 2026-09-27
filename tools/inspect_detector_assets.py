"""Read-only inventories for the isolated detector experiment."""
import json
import urllib.request
import zipfile
from pathlib import Path

for path in Path('/workspace/data/benchmarks/detector-training/archives').glob('*.zip'):
    with zipfile.ZipFile(path) as archive:
        print(path.name, len(archive.namelist()), archive.namelist()[:12], flush=True)
        for name in archive.namelist():
            if name.endswith(('.yaml', 'classes.txt', '.names')):
                print(name, archive.read(name)[:3000].decode(errors='replace'),flush=True)
        text = [n for n in archive.namelist() if n.endswith('.txt')]
        for name in text[:2]:
            print(name,archive.read(name)[:1000].decode(errors='replace'),flush=True)
with urllib.request.urlopen('https://api.github.com/repos/ankandrew/open-image-models/releases/tags/assets', timeout=30) as response:
    release = json.load(response)
    print(json.dumps([{'name':a['name'],'bytes':a['size'],'url':a['browser_download_url'],'digest':a.get('digest')}
                      for a in release['assets'] if a['name'].endswith('.pt')]),flush=True)
