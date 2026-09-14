import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from formal_evaluation.run_ablation_seven_tables import reusable_rows


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


if __name__=='__main__':unittest.main()
