"""Render local publisher-label spot checks; not independent ground truth."""
from pathlib import Path
import cv2
import numpy as np
from sentinel_edge.detector_training_data import read_sources

root = Path('/workspace/data/benchmarks/detector-training')
target = Path('/workspace/data/benchmarks/reports/detector-label-review.jpg')
canvas = np.full((720,1200,3),245,np.uint8)
slot = 0
for kind in ('vehicle','plate'):
    rows,_ = read_sources(root,kind)
    classes = range(5) if kind=='vehicle' else range(1)
    for cls in classes:
        row = next(r for r in rows if any(b[0]==cls for b in r['boxes']))
        image = cv2.imdecode(np.frombuffer(row['raw'],np.uint8),cv2.IMREAD_COLOR)
        for c,x,y,w,h in row['boxes']:
            if c != cls: continue
            box = (round((x-w/2)*row['width']),round((y-h/2)*row['height']),
                   round((x+w/2)*row['width']),round((y+h/2)*row['height']))
            cv2.rectangle(image,box[:2],box[2:],(0,0,255),2)
        h,w = image.shape[:2]; scale = min(390/w,320/h)
        tile = cv2.resize(image,(round(w*scale),round(h*scale)))
        xx,yy = slot%3*400,slot//3*360
        canvas[yy+30:yy+30+tile.shape[0],xx:xx+tile.shape[1]] = tile
        cv2.putText(canvas,f'{kind} class {cls}',(xx+5,yy+24),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,0),1)
        slot += 1
cv2.imwrite(str(target),canvas)
print(target,flush=True)
