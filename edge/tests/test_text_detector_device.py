"""The CPU text helper must not request a missing legacy Paddle memory pass."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

from sentinel_edge.text_boundaries import TextBoundaryDetector


class PredictorConfig:
    def __init__(self,*_):self.memory=False;self.gpu=False
    def disable_glog_info(self):pass
    def enable_memory_optim(self):self.memory=True
    def enable_use_gpu(self,*_):self.gpu=True
    def disable_gpu(self):self.gpu=False
    def set_cpu_math_library_num_threads(self,*_):pass


class TextDetectorDeviceTests(unittest.TestCase):
    def build(self,device):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            assets=[]
            for name in ('inference.json','inference.pdiparams'):
                (root/name).write_bytes(b'test-asset');assets.append(dict(file=name,sha256=hashlib.sha256(b'test-asset').hexdigest()))
            (root/'receipt.json').write_text(json.dumps(dict(repository='PaddlePaddle/PP-OCRv5_mobile_det',
                revision='0d63e78e2b680928f6b1747d76a08db6e645efb7',assets=assets)))
            configs=[]
            def create(config):
                if config.memory and not config.gpu:raise ValueError('Not find predictor_id and memory_optimize_pass')
                configs.append(config);return object()
            paddle=SimpleNamespace(device=SimpleNamespace(is_compiled_with_cuda=lambda:True),
                inference=SimpleNamespace(Config=PredictorConfig,create_predictor=create))
            post=ModuleType('ppocr.postprocess.db_postprocess');post.DBPostProcess=lambda **_:None
            modules={'paddle':paddle,'ppocr':ModuleType('ppocr'),'ppocr.postprocess':ModuleType('ppocr.postprocess'),
                     'ppocr.postprocess.db_postprocess':post}
            before=list(sys.path)
            try:
                with patch.dict(sys.modules,modules):TextBoundaryDetector(root,framework=root,device=device)
            finally:sys.path[:]=before
            return configs[0]

    def test_cpu_predictor_initializes_without_the_missing_memory_pass(self):
        config=self.build('cpu');self.assertFalse(config.memory);self.assertFalse(config.gpu)

    def test_gpu_configuration_keeps_its_existing_memory_setting(self):
        config=self.build('gpu');self.assertTrue(config.memory);self.assertTrue(config.gpu)


if __name__=='__main__':unittest.main()
