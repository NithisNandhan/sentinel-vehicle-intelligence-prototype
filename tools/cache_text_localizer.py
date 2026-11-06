"""Fetch only the pinned ~5 MB official text detector, never a recognizer."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = 'PaddlePaddle/PP-OCRv5_mobile_det'
REVISION = '0d63e78e2b680928f6b1747d76a08db6e645efb7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    assets=[]
    for name in ('inference.json','inference.pdiparams','inference.yml','README.md'):
        url=f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}'
        target=args.output/name
        if target.exists():
            raise FileExistsError('Existing detector assets preserved')
        with urllib.request.urlopen(url,timeout=60) as response:
            content=response.read(8*1024*1024+1)
        if len(content)>8*1024*1024:
            raise ValueError('Lightweight detector download exceeded bound')
        target.write_bytes(content)
        assets.append({'file':name,'url':url,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    receipt={'repository':REPOSITORY,'revision':REVISION,'licence':'Apache-2.0','purpose':'Text boundaries only; no character recognizer', 'assets':assets}
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'assets':len(assets),'bytes':sum(a['bytes'] for a in assets)}))


if __name__=='__main__':
    main()
