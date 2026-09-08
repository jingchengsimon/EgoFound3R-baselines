"""Batch export must preserve the B axis and never mix clips."""
import ast,dataclasses
from pathlib import Path
import unittest

class Tensor:
    def __init__(self,values):self.values=values;self.shape=(len(values),2)
    def __getitem__(self,key):return Tensor(self.values[key])

@dataclasses.dataclass
class FrameMap:
    global_anchor_indices:Tensor
    note:object=None

class BatchExportTest(unittest.TestCase):
    def test_batch_axis_and_map_are_sliced_together(self):
        path=Path(__file__).parents[1]/'run_egofound3r_resident.py'
        tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='batch_slices')
        namespace={'dataclasses':dataclasses,'torch':type('Torch',(),{'Tensor':Tensor})}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
        keys=next(n.value for n in fn.body if isinstance(n,ast.Assign) and n.targets[0].id=='keys')
        outputs={k:Tensor([0,1,2]) for k in ast.literal_eval(keys)}
        sliced,mapping=namespace['batch_slices'](outputs,FrameMap(Tensor([10,11,12])),1,3)
        self.assertTrue(all(v.values==[1] for v in sliced.values()))
        self.assertEqual(mapping['multirate_frame_map'].global_anchor_indices.values,[11])
        self.assertTrue(all(v.values==[0,1,2] for v in outputs.values()))
        with self.assertRaises(AssertionError):namespace['batch_slices'](outputs,FrameMap(Tensor([10,11,12])),1,2)

if __name__=='__main__':unittest.main()
