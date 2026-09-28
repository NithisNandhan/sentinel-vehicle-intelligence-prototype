"""Protected, read-only live counters. Never print media, text or credentials."""
import json
import os
import re
import urllib.request

BASE = 'http://127.0.0.1:7780'
page = urllib.request.urlopen(BASE+'/',timeout=10).read().decode()
match = re.search(r'const viewerToken=("[^"\n]+")',page)
if match is None:
    raise RuntimeError('Viewer token not found')
token = json.loads(match[1])
request = urllib.request.Request(BASE+'/engineering',headers={
    'X-Viewer-Token':token,'X-Operator-Key':os.environ['OPERATOR_API_KEY']})
data = json.load(urllib.request.urlopen(request,timeout=15))
fields = ('status','failure_code','camera_id','decoded_frames','analysed_frames','vehicle_box_instances',
          'plate_crops','plate_detector_calls','targeted_searches','ocr_calls',
          'distinct_ocr_observations','maximum_distinct_observations',
          'unique_model_accepted_unverified','frame_age_seconds','processing_ms')
print(json.dumps({key:data.get(key) for key in fields}),flush=True)
