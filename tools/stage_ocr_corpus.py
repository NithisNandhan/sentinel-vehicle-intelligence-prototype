"""Hash-checked small training export, excluding original archives and live feeds."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extract',action='store_true')
    parser.add_argument('--sha256')
    args=parser.parse_args()
    if args.extract:
        if args.root != Path('/models/experiments/corpus-stage') or not args.sha256:
            raise ValueError('Explicit isolated staging root and checksum required')
        if hashlib.sha256(args.output.read_bytes()).hexdigest()!=args.sha256:
            raise ValueError('Staging archive checksum mismatch')
        with tarfile.open(args.output) as bundle:
            members=bundle.getmembers()
            if len(members)>2001 or sum(m.size for m in members)>512*1024*1024:
                raise ValueError('Staging extraction exceeds bounds')
            for member in members:
                path=(args.root/member.name).resolve()
                if not member.isfile() or not path.is_relative_to(args.root.resolve()):
                    raise ValueError('Unsafe staging member')
                path.parent.mkdir(parents=True,exist_ok=True)
                with bundle.extractfile(member) as source, path.open('wb') as destination:
                    destination.write(source.read())
        print(json.dumps({'export_extracted':len(members)-1,'root':str(args.root)}))
        raise SystemExit(0)
    manifest=args.root/'adaptation_manifest.json'
    rows=json.loads(manifest.read_text())['samples']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(args.output,'w') as bundle:
        bundle.add(manifest,arcname='adaptation_manifest.json')
        for row in rows:
            path=(args.root/row['crop_path']).resolve()
            if not path.is_relative_to(args.root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['crop_sha256']:
                raise ValueError('Unsafe or modified staging crop')
            bundle.add(path,arcname=row['crop_path'])
    print(json.dumps({'staged_crops':len(rows),'archive_bytes':args.output.stat().st_size,'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest()}))
