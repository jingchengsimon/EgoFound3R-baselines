"""Run with: python -m unittest formal_evaluation.tests.test_2d_overlap"""
import unittest
import numpy as np
from formal_evaluation.select_2d_overlap_worker import mask, measures

class SilhouetteTest(unittest.TestCase):
    def test_overlap_and_empty_support(self):
        v=np.array([[2.,2.,1.],[12.,2.,1.],[2.,12.,1.]])
        faces=np.array([[0,1,2]])
        a=mask(v,np.eye(3),faces,(32,32))
        same=measures(a,a)
        self.assertEqual(same,dict(iou=1.,boundary_f=1.,center_error=0.,area_error=0.))
        b=mask(v+np.array([15,15,0]),np.eye(3),faces,(32,32))
        shifted=measures(a,b)
        self.assertEqual(shifted['iou'],0.)
        self.assertEqual(shifted['boundary_f'],0.)
        self.assertGreater(shifted['center_error'],0.)
        self.assertIsNone(measures(np.zeros_like(a),a))
        with self.assertRaisesRegex(ValueError,'NEAR_PLANE'):
            mask(v*np.array([1,1,-1]),np.eye(3),faces,(32,32))

if __name__=='__main__':unittest.main()
