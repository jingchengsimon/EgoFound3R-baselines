"""Regression checks for frozen contact distance reduction and report transport."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from formal_evaluation.aggregate_contact_distance import window_metrics
from formal_evaluation import taskctl


class ContactDistanceTests(unittest.TestCase):
    def test_mask_units_and_undefined_contact(self):
        arrays={}
        for granularity,n in [('joint',21),('marker',195),('vertex',778)]:
            d=np.full((2,2,n),.01);d[1]=1.
            arrays[granularity+'_contact_distance']=d
            arrays[granularity+'_contact_distance_mask']=np.ones(d.shape,bool)
            arrays[granularity+'_contact_probability']=np.ones(d.shape)
        result=window_metrics(arrays,np.array([True,False]))
        for granularity in ('joint','marker','vertex'):
            self.assertAlmostEqual(result[granularity+'_contact_distance_all_valid_mm'],10.)
        arrays['joint_contact_probability'][:]=0
        result=window_metrics(arrays,np.array([True,False]))
        self.assertTrue(np.isnan(result['joint_contact_distance_predicted_contact_mm']))
        self.assertEqual(result['joint_contact_distance_predicted_contact_count'],0)

    def test_registered_non_methods_report_is_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'report.json'
            datasets={'h2o':{'n_windows':12,'joint_contact_distance_all_valid_mm_mean':12.3}}
            report.write_text(json.dumps({'method':'interactvlm','windows':100,'datasets':datasets}))
            spec={'key':'test','output_root':directory,'completion_artifacts':[{'path':str(report),'min_bytes':1}]}
            result=subprocess.run([sys.executable,'-c',taskctl.REMOTE_PROGRESS_PROBE,json.dumps([spec])],capture_output=True,text=True,check=True)
            returned=json.loads(result.stdout)['test']['report_summary']['reports'][str(report)]
            self.assertEqual(returned['datasets'],datasets)

if __name__=='__main__':unittest.main()
