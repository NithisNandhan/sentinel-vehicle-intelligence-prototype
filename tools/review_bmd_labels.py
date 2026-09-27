"""Render a small public BMD annotation spot check, not independent truth."""
import json
from pathlib import Path

import cv2
import numpy as np

root=Path('/training/vehicle-bmd')
manifest=json.loads((root/'manifest.json').read_text())
canvas=np.full((750,1500,3),240,np.uint8)
samples=[]
for split in ('train','val','test'):
    samples.extend([s for s in manifest['samples'] if s['split']==split][:2])
for index,row in enumerate(samples):
    image=cv2.imread(str(root/row['image_path']))
    height,width=image.shape[:2]
    for cls,x,y,w,h in row['boxes']:
        x1,y1=round((x-w/2)*width),round((y-h/2)*height)
        x2,y2=round((x+w/2)*width),round((y+h/2)*height)
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
    tile=cv2.resize(image,(500,281))
    xx,yy=index%3*500,index//3*375
    canvas[yy+35:yy+316,xx:xx+500]=tile
    cv2.putText(canvas,row['id']+' '+row['split'],(xx+8,yy+25),
                cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
target=Path('/reports/bmd-label-review.jpg')
cv2.imwrite(str(target),canvas)
print(str(target),flush=True)
