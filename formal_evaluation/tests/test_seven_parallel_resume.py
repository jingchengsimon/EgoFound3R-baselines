import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from formal_evaluation.run_ablation_seven_tables import reusable_rows
from formal_evaluation.run_ablation_seven_parallel import materialize_resume


class ResumeTests(unittest.TestCase):
    def test_complete_records_only_and_disjoint_eight_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'h2o';p.mkdir()
            gt={str(i):{'cache_id':str(i),'frame_ids':[0,1]} for i in range(19)}
            rows=[{'window_id':str(i),'frame_ids':[0,1]} for i in range(3)]
            (p/'window_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows)+'{"window_id":')
            with zipfile.ZipFile(p/'0_scene.npz','w') as z:
                for i in range(6):z.writestr(str(i),b'x')
            (p/'1_scene.npz').write_bytes(b'incomplete zip')
            reused=reusable_rows(tmp,'h2o',gt)
            self.assertEqual(set(reused),{'0'})
            shards=[[w for i,w in enumerate(sorted(gt)) if i%8==s and w not in reused] for s in range(8)]
            flat=sum(shards,[])
            self.assertEqual(len(flat),len(set(flat)))
            self.assertEqual(set(flat)|set(reused),set(gt))
            rows[0]['frame_ids']=[1,0]
            (p/'window_metrics.jsonl').write_text(json.dumps(rows[0])+'\n')
            with self.assertRaises(ValueError):reusable_rows(tmp,'h2o',gt)


    def test_sixteen_workers_resume_multiple_roots_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);gt=[dict(window_id=str(i),cache_id=str(i),frame_ids=[0,1]) for i in range(37)]
            index=base/'gt.jsonl';index.write_text(''.join(json.dumps(r)+'\n' for r in gt))
            selection=base/'selection.jsonl';selection.write_text(''.join(json.dumps(dict(r,dataset='h2o'))+'\n' for r in gt))
            roots=[]
            for i in range(3):
                root=base/str(i);folder=root/'h2o';folder.mkdir(parents=True);roots.append(str(root))
                (folder/'window_metrics.jsonl').write_text(json.dumps(gt[i])+'\n')
                with zipfile.ZipFile(folder/(str(i)+'_scene.npz'),'w') as z:
                    for k in range(6):z.writestr(str(k),b'x')
            spec=dict(workers=16,resume_roots=roots,selection=str(selection),selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),jobs=[dict(dataset='h2o',gt_index=str(index),expected_windows=37)])
            report=materialize_resume(spec,base/'snapshot')['h2o'];flat=sum(report['shards'],[])
            self.assertEqual(report['reused'],3);self.assertEqual(len(flat),34)
            self.assertEqual(len(set(flat)),34);self.assertTrue(set(flat).isdisjoint({'0','1','2'}))
            self.assertEqual(set(reusable_rows(base/'snapshot','h2o',{r['window_id']:r for r in gt})),{'0','1','2'})
            spec['resume_roots'].append(roots[0])
            with self.assertRaisesRegex(ValueError,'duplicate resume window'):materialize_resume(spec,base/'duplicate')


if __name__=='__main__':unittest.main()
