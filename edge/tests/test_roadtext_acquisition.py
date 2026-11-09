import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[2]/'tools/acquire_roadtext3k.py'
        spec=importlib.util.spec_from_file_location('roadtext_acquirer',path)
        cls.tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.tool)

    def test_sparse_video_index_uses_next_existing_key(self):
        payload=json.dumps({str(i):{'1':{'labels':[]},'2':{'labels':[]}} for i in (1,3,7,9)}).encode()
        metadata=dict(bytes=len(payload),etag='fixture')
        def read(url,start,end,meta):return payload[start:end+1]
        with patch.object(self.tool,'read_range',side_effect=read), patch('builtins.print'):
            for target,expected in ((1,1),(2,3),(3,3),(4,7),(9,9)):
                offset=self.tool.locate_video('fixture',target,metadata)
                self.assertTrue(payload[offset:].startswith(('"'+str(expected)+'":').encode()))


if __name__=='__main__':unittest.main()
