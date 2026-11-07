import collections
import pathlib
import tempfile
import unittest
import os
import json
import math
from bench.run import balanced_blocks, first_mismatch
from bench.common import atomic_json, sha256
from analysis.report import cells, paired_summary, summarize
from bench.replay import fit_validate

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

    def test_clustered_statistics_and_correctness_gate(self):
        rows=[]
        for i in range(6):
            for cap in (128,512):
                for rep in range(3):
                    for conf,factor in [('target',1),('fixed2',2),('fixed4',1.5)]:
                        rows.append(dict(prompt_id=f'development-qa-{i:02d}',output_cap=cap,repetition=rep,
                            configuration_id=conf,correctness='pass',generated_tokens=10,generation_ms=100/factor,
                            category='qa',warm_ms=200,load_ms=50,peak_rss_bytes=1024))
        table=cells(rows);result=paired_summary(table,'fixed2','target',samples=200)
        self.assertAlmostEqual(result['ratio'],2)
        self.assertEqual(result['prompts'],6);self.assertEqual(result['prompt_cap_cells'],12)
        self.assertAlmostEqual(result['ci95'][0],2)
        summary=summarize(rows,samples=200);self.assertEqual(summary['selected_fixed'],'fixed2')
        self.assertTrue(summary['q1_supported'])
        rows[0]['correctness']='pending'
        with self.assertRaises(ValueError):cells(rows)

    def test_replay_prediction_separates_fit_and_validation(self):
        records=[]
        for i in range(6):
            for context in (128,512):
                outcomes={str(k):{'rate':.01,'ms':100,'tokens':1} for k in (0,1,2,4,6,8)}
                winner='2' if context<256 else '8'
                outcomes[winner]={'rate':.5,'ms':10,'tokens':5}
                records.append(dict(prompt_id=f'development-qa-{i:02d}',context=context,acceptance=.8,
                    outcomes=outcomes,probe_ms=1,checkpoint=context))
        result=fit_validate(records,0)
        self.assertTrue(result['q2_supported'])
        self.assertEqual(result['validation_prompts'],3)
        for r in records:
            if int(r['prompt_id'].rsplit('-',1)[1])>=3:
                r['outcomes']['2'],r['outcomes']['8']=r['outcomes']['8'],r['outcomes']['2']
        self.assertFalse(fit_validate(records,0)['q2_supported'])

    def test_frozen_prompt_split_and_provenance(self):
        from bench.make_prompts import build
        rows=build()
        self.assertEqual(len(rows),130)
        self.assertEqual(collections.Counter(r['split'] for r in rows),{'development':40,'heldout':90})
        self.assertEqual(collections.Counter(r['category'] for r in rows if r['split']=='heldout'),{'qa':30,'code':30,'summary':30})
        self.assertEqual(sum(r['pilot'] for r in rows),18)
        self.assertEqual(len({r['sha256'] for r in rows}),130)

    @unittest.skipUnless(os.getenv('TOKENOMICS_TEST_BINARY'),'CMake supplies runtime binary for integration')
    def test_incremental_resume_and_fingerprint(self):
        from bench.run import run
        exe=pathlib.Path(os.environ['TOKENOMICS_TEST_BINARY'])
        model=exe.parent/'fixtures/target.gguf'
        cfg={'output_caps':[8], 'repetitions':1,'seed':5,'context':64,'batch':32,'threads':1,'warmup_tokens':2,
             'configurations':[{'id':'target','policy':'target'},{'id':'fixed2','policy':'fixed','k':2}]}
        prompts=[{'id':'development-qa-00','category':'qa','prompt':'a'}]
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)
            rows=run(exe,model,model,prompts,cfg,out,identity={'synthetic_fixture':True})
            self.assertEqual(len(rows),2)
            done=next(out.glob('blocks/*/complete.json'));mtime=done.stat().st_mtime_ns
            self.assertEqual(run(exe,model,model,prompts,cfg,out,identity={'synthetic_fixture':True}),rows)
            self.assertEqual(done.stat().st_mtime_ns,mtime)
            cfg['threads']=2
            with self.assertRaisesRegex(RuntimeError,'Resume refused'):run(exe,model,model,prompts,cfg,out,identity={'synthetic_fixture':True})

    @unittest.skipUnless(os.getenv('TOKENOMICS_TEST_BINARY'),'CMake supplies runtime binary for integration')
    def test_failed_correctness_cannot_commit_block(self):
        from bench.run import run
        import sys
        with tempfile.TemporaryDirectory() as d:
            root=pathlib.Path(d);fake=root/'fake-engine'
            fake.write_text('#!'+sys.executable+'\n'+'''import json,sys
a=dict(zip(sys.argv[1::2],sys.argv[2::2]))
rows=[]
for line in open(a['--jobs']):
 j=json.loads(line);j.update(prompt_ids=[1],token_ids=[1 if j['configuration_id']=='target' else 2],stop='budget');rows.append(j)
open(a['--out'],'w').write(''.join(json.dumps(j)+'\\n' for j in rows))
open(a['--cycles'],'w').close()
''')
            fake.chmod(0o755);model=root/'model';model.write_text('fake')
            cfg={'output_caps':[8],'repetitions':1,'seed':1,'context':64,'batch':32,'threads':1,'warmup_tokens':2,
                 'configurations':[{'id':'target','policy':'target'},{'id':'fixed2','policy':'fixed','k':2}]}
            with self.assertRaisesRegex(RuntimeError,'token_mismatch'):
                run(fake,model,model,[{'id':'development-qa-00','category':'qa','prompt':'a'}],cfg,root/'run')
            self.assertFalse(list((root/'run').glob('blocks/*/complete.json')))
            self.assertTrue(list((root/'run').glob('blocks/*/attempt-*/failure.json')))
