from context import miniball as mb

import unittest
import numpy as np

class TestTypes(unittest.TestCase):
    valid_ftypes = {
                     float:         float,
                     np.float32:    np.float32,
                     np.float64:    np.float64,
                     np.float128:   np.float128,
                     # np.float16:  float, # not available
                     np.float:      float
                    }

    valid_itypes = { int:           float,
                     np.int8:       float,
                     np.int16:      float,
                     np.int32:      float,
                     np.int64:      float,
                     np.uint:       float,
                     np.uint8:      float,
                     np.uint16:     float,
                     np.uint32:     float,
                     np.uint64:     float,
                     #np.bool8:     float, # supported, but not tested.
                     #np.bool:      float  # supported, but not tested.
                   }
    invalid_dtypes = [ str, np.str, np.complex, np.complex64,
                       np.complex128, np.complex256, np.float16 ]
    iterables = [list, tuple, set]
    detailed = False

    def checkRet(self, ret, ref, dtype):
        cRef, r2Ref = ref
        cRet, r2Ret = ret[0:2]
        # Strangely, this is always float, independent of the requested dtype.
        # TODO: Check why.
        self.assertTrue(issubclass(type(r2Ret), float))
        self.assertEqual(cRet.dtype, dtype)
        if cRef is not None:
            np.testing.assert_array_equal(cRet, cRef)
        else:
            self.assertIsNone(cRet)
        self.assertEqual(r2Ret, r2Ref)
        if self.detailed:
            self.assertEqual(len(ret), 3)
            info = ret[2]
            if cRef is not None:
                np.testing.assert_array_equal(info["center"], cRef)
                self.assertEqual(info["radius"], np.sqrt(r2Ref))
            else:
                self.assertIsNone(info)

    def checkRetCC(self, ret, ref):
        # Handles also nans.
        #self.assertEqual(ret[:2], ref[:2])
        np.testing.assert_equal(ret[:2], ref[:2])
        if self.detailed:
            info = ret[2]
            infoRef = ref[2]
            if info is not None:
                self.assertSubset(info, infoRef)

    def assertSubset(self, dictRet, dictRef):
        dictRet = {k:v for k,v in dictRef.items() if k in dictRet}
        self.assertEqual(dictRet, dictRef)

    def convertData(self, data, dtype):
        return [list(map(dtype, elm)) for elm in data]

    def testNumpyTypes(self):
        # All values positive because of unsigned int tests.
        dataIn = [[7.,7.], [2.,6.], [1.,1.], [3., 0.]]
        ref = ([4., 4.], 18.0)
        dtypes = {**self.valid_ftypes, **self.valid_itypes}
        for dt, dt_ret in dtypes.items():
            with self.subTest(dt=dt):
                data = self.convertData(dataIn, dt)
                ret = mb.compute(data, details=self.detailed)
                self.checkRet(ret, ref, dtype=dt_ret)
                ret = mb.compute(np.array(data, dtype=dt),
                                 details=self.detailed)
                self.checkRet(ret, ref, dtype=dt_ret)

    def testNumpyTypesNaN(self):
        # Points containing np.nan are skipped.
        # All values positive because of unsigned int tests.
        dataIn = [[7.,np.nan], [2.,6.], [1.,1.], [4, 0]]
        ref = ([3., 3.], 10.0)
        for dt, dt_ret in self.valid_ftypes.items():
            with self.subTest(dt=dt):
                data = self.convertData(dataIn, dt)
                ret = mb.compute(data, details=self.detailed)
                self.checkRet(ret, ref, dtype=dt_ret)
                ret = mb.compute(np.array(data, dtype=dt),
                                 details=self.detailed)
                self.checkRet(ret, ref, dtype=dt_ret)

    def testInvalidNumpyTypes(self):
        data = [1,2,3]
        for dt in self.invalid_dtypes:
            with self.subTest(dt=dt):
                d = np.array(data, dtype=dt)
                self.assertRaises(mb.MiniballTypeError, mb.compute, d)

    def testConversionFromIterable(self):
        data = [(4.,4.), (-1.,3.), (-2.,-2.), (1, -3)] + [(2.,1.)]*20
        ref = ([1., 1.], 18.0)
        for cls in self.iterables:
            with self.subTest(cls=cls):
                d = cls(data)
                ret = mb.compute(d, details=self.detailed)
                self.checkRet(ret, ref, dtype=float)

    def testSlices(self):
        n = 21
        d = 6
        t = np.linspace(0,1,n)
        a = np.array([-2]*d)[:,np.newaxis]
        b = np.array([ 2]*d)[:,np.newaxis]
        #d = np.ascontiguousarray(((b-a)*t+a).T)
        d = ((b-a)*t+a).T
        D1 = d
        D2 = d[::4,::2]
        refAll = (np.zeros(D1.shape[1]), 24)
        refSlice = (np.zeros(D2.shape[1]), 12)
        retAll = mb.compute(D1, details=self.detailed)
        retSlice = mb.compute(D2, details=self.detailed)
        self.checkRet(retAll, refAll, dtype=float)
        self.checkRet(retSlice, refSlice, dtype=float)

    def testCornerCases(self):
        # None; results in (None, 0)
        d = None
        ret = (None, 0, None)
        self.checkRetCC(mb.compute(d, details=self.detailed), ret)
        # Empty array; results in (None, 0)
        d = []
        ret = (None, 0, None)
        self.checkRetCC(mb.compute(d, details=self.detailed), ret)
        self.checkRetCC(mb.compute(np.array(d, dtype=int),
                                   details=self.detailed), ret)
        # Scalar; is treated as one 1D point [42].
        d = 42
        ret = ([42], 0, dict(center=42, radius=0, n_support=1, support=[0]))
        self.checkRetCC(mb.compute(d, details=self.detailed), ret)
        self.checkRetCC(mb.compute(np.array(d, dtype=int),
                                   details=self.detailed), ret)
        # 1D array; is treated as a list of 1D points.
        d = [1,2,4,5]
        ret = ([3], 4, dict(center=3, radius=4, n_support=2, support=[0,3]))
        self.checkRetCC(mb.compute(d, details=self.detailed), ret)
        self.checkRetCC(mb.compute(np.array(d, dtype=int),
                                   details=self.detailed), ret)
        # 3D array; raises a RuntimeError.
        d = [[[1,2,3], [4,5,6]]]
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballTypeError, mb.compute, np.array(d))
        # String; raises a RuntimeError.
        d = "abc"
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballTypeError, mb.compute, np.array(d))
        # Complex; raises a RuntimeError.
        d = [[1+2j, 3+4j], [5+6j, 7+8j]]
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballTypeError, mb.compute, np.array(d))
        # [None]; raises a RuntimeError.
        d = [None]
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballTypeError, mb.compute, np.array(d))
        # Mixed container; raises a RuntimeError.
        d = [set(), 1, "abc"]
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballTypeError, mb.compute, np.array(d))
        # Mixed container; raises a RuntimeError.
        d = [2, 1, None]
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballTypeError, mb.compute, np.array(d))
        # All nans; equivalent with empty array.
        d = [[np.nan, np.nan], [1, np.nan], [0, np.nan]]
        ret = (None, np.nan, dict(center=None, support=None, is_valid=False))
        self.checkRetCC(mb.compute(d, details=self.detailed), ret)
        self.checkRetCC(mb.compute(np.array(d),
                                   details=self.detailed), ret)

class TestTypesDetailed(TestTypes):
    detailed = True

################################################################################
class TestRandom(unittest.TestCase):
    n_points = [0,1,10,10000]
    n_dims = [0,1,2,3]

    def setUp(self):
        self.rs = np.random.RandomState(42)

    def testPlain(self):
        for d in self.n_dims:
            for n in self.n_points:
                with self.subTest(n=n, d=d):
                    size = (n, d) if d else (n,)
                    data = self.rs.normal(0,1,size)
                    C_A, r2_A = mb.compute(data, details=False)
                    C_B, r2_B, info_B = mb.compute(data, details=True)
                    np.testing.assert_array_equal(C_A, C_B)
                    self.assertEqual(r2_A, r2_B)

################################################################################
if __name__ == "__main__":
    unittest.main(verbosity=2)
