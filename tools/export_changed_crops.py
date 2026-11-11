"""Export native before/after images for a provenance-bound crop-only replay."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from render_crop_comparison import load,safe_file,sha


def export(baseline,current,output):
    baseline,current,output=map(Path,(baseline,current,output))
    old,new=load(baseline/'predictions.json'),load(current/'predictions.json')
    if new.get('crop_replay',{}).get('input_predictions_sha256')!=sha(baseline/'predictions.json'):
        raise ValueError('Only a replay bound to this baseline may reuse plate IDs')
    if old['source_sha256']!=new['source_sha256']:raise ValueError('Different source videos')
    previous={g['id']:g for g in old['groups']};changed=[]
    for group in new['groups']:
        before=previous[group['id']]
        for a,b in zip(before['samples'],group['samples']):
            if a['source_id']!=b['source_id']:changed.append((before,group,a,b));break
    if not changed:raise ValueError('No changed crop histories')
    if len(changed)>18:raise ValueError('Bounded contact sheet supports up to eighteen changed histories')
    canvas=Image.new('RGB',(1230,95+len(changed)*245),'#f7f7f1');draw=ImageDraw.Draw(canvas)
    font=ImageFont.load_default(size=18);small=ImageFont.load_default(size=15)
    draw.text((20,18),'Changed native crops | previous vs new',font=font,fill='#0F766E')
    draw.text((20,48),'Readings are unverified unless covered by your human labels; this image does not add accuracy targets.',font=small,fill='#526660')
    for row,(before,after,a,b) in enumerate(changed):
        y=95+row*245
        for column,(group,sample,root,label) in enumerate(((before,a,baseline,'Previous'),(after,b,current,'New'))):
            x=20+column*605
            draw.text((x,y),f"{label} | {group['id']} | {sample['position_ms']/1000:.3f}s",font=font,fill='#0F766E')
            draw.rectangle((x,y+30,x+580,y+185),fill='#e9eee8')
            with Image.open(safe_file(root,sample['path'])) as source:
                image=source.convert('RGB')
                import hashlib
                if hashlib.sha256(image.tobytes('raw','BGR')).hexdigest()!=sample['source_id']:
                    raise ValueError('Native source pixels changed')
                scale=min(2,560/image.width,145/image.height)
                image=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
                canvas.paste(image,(x+(580-image.width)//2,y+30+(155-image.height)//2))
            draw.text((x,y+195),group['methods']['reconstruction']['top'] or 'No reading',font=font,fill='#193b36')
    if output.exists():raise ValueError('Preserve existing comparison exports; choose a fresh filename')
    if not output.resolve().is_relative_to(Path(__file__).resolve().parents[1]/'data/benchmarks'):
        raise ValueError('Image evidence must stay in local benchmark data')
    canvas.save(output)
    return len(changed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','current','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(dict(changed_histories=export(args.baseline,args.current,args.output))))
