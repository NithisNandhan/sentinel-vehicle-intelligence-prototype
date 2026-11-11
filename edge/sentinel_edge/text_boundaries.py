"""Cached, hash-checked text detection only; never predicts plate characters.

Text proposals can include branding and signs. They are not verified plates and
must not enter the sighting/alert path without an independent plate check.
"""
import hashlib
import json
from pathlib import Path
import sys


class TextBoundaryDetector:
    def __init__(self, assets, *, framework=Path('/models/experiments/awiros/PaddleOCR'), device='gpu'):
        if device not in ('cpu','gpu'):
            raise ValueError('Explicit text-detector device required')
        assets = Path(assets)
        receipt = json.loads((assets/'receipt.json').read_bytes())
        if receipt['repository'] != 'PaddlePaddle/PP-OCRv5_mobile_det' or receipt['revision'] != '0d63e78e2b680928f6b1747d76a08db6e645efb7':
            raise ValueError('Unexpected text-detector asset revision')
        for row in receipt['assets']:
            path=(assets/row['file']).resolve()
            if not path.is_relative_to(assets.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('Text detector asset changed')
        import paddle
        if device=='gpu' and not paddle.device.is_compiled_with_cuda():
            raise RuntimeError('GPU text detection unavailable')
        sys.path.insert(0,str(framework))
        from ppocr.postprocess.db_postprocess import DBPostProcess
        config=paddle.inference.Config(str(assets/'inference.json'),str(assets/'inference.pdiparams'))
        config.disable_glog_info()
        if device=='gpu':
            config.enable_memory_optim()
            config.enable_use_gpu(256,0)
        else:
            # Paddle's CPU PIR predictor has no memory_optimize_pass result;
            # enabling that legacy pass raises at create_predictor().
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(2)
        self.predictor=paddle.inference.create_predictor(config)
        self.postprocess=DBPostProcess(thresh=.3,box_thresh=.6,max_candidates=1000,unclip_ratio=1.5)
        self.device=device

    def polygons(self, image):
        import cv2
        import numpy as np
        h,w=image.shape[:2]
        if not h or not w:
            return []
        ratio=960/max(h,w)
        ih,iw=max(32,round(h*ratio/32)*32),max(32,round(w*ratio/32)*32)
        resized=cv2.resize(image,(iw,ih)).astype(np.float32)/255.
        tensor=((resized-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        handle=self.predictor.get_input_handle(self.predictor.get_input_names()[0])
        handle.reshape(tensor.shape)
        handle.copy_from_cpu(np.ascontiguousarray(tensor))
        self.predictor.run()
        maps=self.predictor.get_output_handle(self.predictor.get_output_names()[0]).copy_to_cpu()
        results=self.postprocess({'maps':maps},np.array([[h,w,ih/h,iw/w]],np.float32))
        return [p.tolist() for p in results[0]['points']]


def warp_polygon(image, polygon):
    import cv2
    import numpy as np
    a,b,c,d=points=np.array(polygon,np.float32)
    w=round(max(np.linalg.norm(b-a),np.linalg.norm(c-d)))
    h=round(max(np.linalg.norm(d-a),np.linalg.norm(c-b)))
    if w<24 or h<8 or w*h>2_000_000:
        return None
    target=np.array([(0,0),(w-1,0),(w-1,h-1),(0,h-1)],np.float32)
    result=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,target),(w,h),flags=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(result,3,3,4,4,cv2.BORDER_REPLICATE)
