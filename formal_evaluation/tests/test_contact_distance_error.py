import unittest
import numpy as np
from formal_evaluation.contact.distance_error import window_distance_errors


class DistanceErrorTest(unittest.TestCase):
    def test_correspondence_subsets_masks_and_units(self):
        pred, gt = {}, {}
        for prefix, n in (("joint",21),("marker",195),("vertex",778)):
            s=prefix+"_contact_"; shape=(2,2,n)
            pred[s+"distance"]=np.full(shape,.1)
            gt[s+"distance"]=np.zeros(shape)
            for a in (pred,gt): a[s+"distance_mask"]=np.zeros(shape,bool); a[s+"distance_mask"][0,0,:3]=True
            pred[s+"distance"][0,0,:3]=[.011,.025,.019]
            gt[s+"distance"][0,0,:3]=[.01,.02,.01]
            pred[s+"probability"]=np.zeros(shape); pred[s+"probability"][0,0,0]=.5
            gt[s+"target"]=np.zeros(shape); gt[s+"target"][0,0,2]=1
            gt[s+"mask"]=np.ones(shape,bool)
        values=window_distance_errors(pred,gt,[True,False])
        for prefix in ("joint","marker","vertex"):
            for subset, expected in (("all",5),("pred_contact",1),("gt_contact",9)):
                self.assertAlmostEqual(values[prefix+"_contact_distance_mae_"+subset+"_mm"],expected)
        values=window_distance_errors(pred,gt,[False,False])
        self.assertTrue(np.isnan(values["joint_contact_distance_mae_all_mm"]))
        self.assertEqual(values["joint_contact_distance_mae_all_count"],0)
