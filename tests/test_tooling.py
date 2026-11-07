import collections
import pathlib
import tempfile
import unittest
from bench.run import balanced_blocks, first_mismatch
from bench.common import atomic_json, sha256

class ToolingTests(unittest.TestCase):
    def test_balanced_order_and_determinism(self):
        prompts=[{'id':str(i)} for i in range(6)]
        confs=[{'id':str(i)} for i in range(3)]
        blocks=list(balanced_blocks(prompts,[128,512],3,confs,123))
        self.assertEqual(blocks,list(balanced_blocks(prompts,[128,512],3,confs,123)))
        self.assertEqual(len(blocks),36)
        counts=collections.Counter((c['id'],slot) for _,_,_,order in blocks for slot,c in enumerate(order))
        self.assertEqual(set(counts.values()),{12})
        self.assertEqual(len({(p['id'],cap,r) for p,cap,r,_ in blocks}),36)

    def test_first_divergence_including_length(self):
        self.assertIsNone(first_mismatch([1,2],[1,2]))
        self.assertEqual(first_mismatch([1,2],[1,3])['position'],1)
        self.assertEqual(first_mismatch([1],[1,2])['position'],1)

    def test_atomic_artifact_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'record.json'
            atomic_json(p,{'x':1}); before=sha256(p)
            atomic_json(p,{'x':2});self.assertNotEqual(before,sha256(p))
            self.assertFalse(p.with_suffix('.json.tmp').exists())
