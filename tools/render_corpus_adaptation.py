"""Before/after licensed still diagnostics; no invented multi-frame acceptance."""
import json
from pathlib import Path
import cv2
from sentinel_edge.awiros_experiment import AwirosExperiment
from sentinel_edge.benchmark_viewer import render_viewer
from sentinel_edge.ctc_fusion import CTCPlateReader, greedy_text

cache=Path('/models/experiments/awiros')
root=Path('/models/experiments/corpus-stage')
report=Path('/workspace/data/benchmarks/reports')
document=json.loads((root/'adaptation_manifest.json').read_text())
rows=[r for r in document['samples'] if r['split']=='test']
visuals=[]
for name,checkpoint in [('Original Awiros',None),('Deeper adapted Awiros',cache/'adaptations/indian-corpus-encoder.safetensors')]:
    reader=AwirosExperiment(cache,device='gpu',checkpoint=checkpoint)
    for i,row in enumerate(rows,1):
        image=cv2.imread(str(root/row['crop_path']))
        probabilities=reader.probabilities(image)
        text=greedy_text(probabilities,reader.tokens)
        fusion=CTCPlateReader(reader.tokens)
        fusion.add(probabilities,timestamp_ms=0,source_id=row['id'])
        visuals.append({'source_mode':'benchmark','diagnostic_only':True,'title':f'Test crop {i} · {name}',
                        'original':image,'candidates':fusion.candidates(),'accepted':None,'expected':row['transcription'],
                        'outcome':'raw-exact' if text==row['transcription'] else 'raw-error',
                        'reason':'Single still diagnostic, no temporal acceptance',
                        'frame_diagnostics':[{'image':image,'index':0,'pts_ms':None,'reading':text,'ocr_ms':'not profiled here'}],
                        'score_note':document['evaluation_caveat']+'; '+row['label_tier']+'. Model scores are not calibrated confidence.'})
        print(json.dumps({'viewer_crop':i,'model':name}),flush=True)
count=len(rows)
visuals=[v for i in range(count) for v in (visuals[i],visuals[i+count])]
dataset={'license':'CC-BY-4.0','permitted_use':'noncommercial_benchmark','name':'Bapatla diagnostic subset',
         'source_url':'https://zenodo.org/records/13954136','license_url':'https://creativecommons.org/licenses/by/4.0/',
         'modifications':'Native crops; no source image enhancement in this viewer'}
render_viewer(report/'indian-corpus-adaptation.html',dataset,visuals)
print(json.dumps({'viewer':'indian-corpus-adaptation.html','test_crops':count,'cards':len(visuals)}),flush=True)
