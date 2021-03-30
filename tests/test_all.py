import unittest
import numpy as np
import cyminiball as mb


class TestTypes(unittest.TestCase):
    valid_ftypes = {float:         float,
                    np.float32:    np.float32,
                    np.float64:    np.float64,
                    np.float128:   np.float128,
                    # np.float16:  float, # not available
                    # np.float:    float  # deprecated
                    }

    valid_itypes = {int:           float,
                    np.int8:       float,
                    np.int16:      float,
                    np.int32:      float,
                    np.int64:      float,
                    np.uint:       float,
                    np.uint8:      float,
                    np.uint16:     float,
                    np.uint32:     float,
                    np.uint64:     float,
                    # np.bool8:    float, # supported, but not tested.
                    # np.bool:     float  # supported, but not tested.
                    }
    invalid_dtypes = [str, complex, np.complex64,
                      np.complex128, np.complex256, np.float16]
    iterables = [list, tuple, set]
    detailed = False

    def check_ret(self, ret, ref, dtype):
        c_ref, r2_ref = ref
        c_ret, r2_ret = ret[0:2]
        # Strangely, this is always float, independent of the requested dtype.
        # TODO: Check why.
        self.assertTrue(issubclass(type(r2_ret), float))
        self.assertEqual(c_ret.dtype, dtype)
        if c_ref is not None:
            np.testing.assert_array_equal(c_ret, c_ref)
        else:
            self.assertIsNone(c_ret)
        self.assertEqual(r2_ret, r2_ref)
        if self.detailed:
            self.assertEqual(len(ret), 3)
            info = ret[2]
            if c_ref is not None:
                np.testing.assert_array_equal(info["center"], c_ref)
                self.assertEqual(info["radius"], np.sqrt(r2_ref))
            else:
                self.assertIsNone(info)

    def check_ret_cc(self, ret, ref):
        # Handles also nans.
        # self.assertEqual(ret[:2], ref[:2])
        np.testing.assert_equal(ret[:2], ref[:2])
        if self.detailed:
            info = ret[2]
            info_ref = ref[2]
            if info is not None:
                self.assert_subset(info, info_ref)

    def assert_subset(self, dict_ret, dict_ref):
        dict_ret = {k: v for k, v in dict_ref.items() if k in dict_ret}
        self.assertEqual(dict_ret, dict_ref)

    def convert_data(self, data, dtype):
        return [list(map(dtype, elm)) for elm in data]

    def test_numpy_types(self):
        # All values positive because of unsigned int tests.
        data_in = [[7., 7.], [2., 6.], [1., 1.], [3., 0.]]
        ref = ([4., 4.], 18.0)
        dtypes = {**self.valid_ftypes, **self.valid_itypes}
        for dt, dt_ret in dtypes.items():
            with self.subTest(dt=dt):
                data = self.convert_data(data_in, dt)
                ret = mb.compute(data, details=self.detailed)
                self.check_ret(ret, ref, dtype=dt_ret)
                ret = mb.compute(np.array(data, dtype=dt),
                                 details=self.detailed)
                self.check_ret(ret, ref, dtype=dt_ret)

    def test_numpy_types_nan(self):
        # Points containing np.nan are skipped.
        # All values positive because of unsigned int tests.
        data_in = [[7., np.nan], [2., 6.], [1., 1.], [4, 0]]
        ref = ([3., 3.], 10.0)
        for dt, dt_ret in self.valid_ftypes.items():
            with self.subTest(dt=dt):
                data = self.convert_data(data_in, dt)
                ret = mb.compute(data, details=self.detailed)
                self.check_ret(ret, ref, dtype=dt_ret)
                ret = mb.compute(np.array(data, dtype=dt),
                                 details=self.detailed)
                self.check_ret(ret, ref, dtype=dt_ret)

    def test_invalid_numpy_types(self):
        data = [1, 2, 3]
        for dt in self.invalid_dtypes:
            with self.subTest(dt=dt):
                d = np.array(data, dtype=dt)
                self.assertRaises(mb.MiniballValueError, mb.compute, d)

    def test_conversion_from_iterable(self):
        data = [(4., 4.), (-1., 3.), (-2., -2.), (1, -3)] + [(2., 1.)]*20
        ref = ([1., 1.], 18.0)
        for cls in self.iterables:
            with self.subTest(cls=cls):
                d = cls(data)
                ret = mb.compute(d, details=self.detailed)
                self.check_ret(ret, ref, dtype=float)

    def test_slices(self):
        n = 21
        d = 6
        t = np.linspace(0, 1, n)
        a = np.array([-2]*d)[:, np.newaxis]
        b = np.array([2]*d)[:, np.newaxis]
        # d = np.ascontiguousarray(((b-a)*t+a).T)
        d = ((b-a)*t+a).T
        D1 = d
        D2 = d[::4, ::2]
        ref_all = (np.zeros(D1.shape[1]), 24)
        ref_slice = (np.zeros(D2.shape[1]), 12)
        ret_all = mb.compute(D1, details=self.detailed)
        ret_slice = mb.compute(D2, details=self.detailed)
        self.check_ret(ret_all, ref_all, dtype=float)
        self.check_ret(ret_slice, ref_slice, dtype=float)

    def test_corner_cases(self):
        # None; results in an exception.
        d = None
        self.assertRaises(mb.MiniballValueError, mb.compute, d)
        # Empty array; results in an exception.
        d = []
        self.assertRaises(mb.MiniballValueError, mb.compute,
                          d, details=self.detailed)
        self.assertRaises(mb.MiniballValueError, mb.compute,
                          np.array(d, dtype=int), details=self.detailed)
        self.assertRaises(mb.MiniballValueError, mb.compute,
                          np.empty([0, 4], dtype=float), details=self.detailed)
        # Scalar; is treated as one 1D point [42].
        d = 42
        ret = ([42], 0, dict(center=42, radius=0, n_support=1, support=[0]))
        self.check_ret_cc(mb.compute(d, details=self.detailed), ret)
        self.check_ret_cc(mb.compute(np.array(d, dtype=int),
                                     details=self.detailed), ret)
        # 1D array; is treated as a list of 1D points.
        d = [1, 2, 4, 5]
        ret = ([3], 4, dict(center=3, radius=4, n_support=2, support=[0, 3]))
        self.check_ret_cc(mb.compute(d, details=self.detailed), ret)
        self.check_ret_cc(mb.compute(np.array(d, dtype=int),
                                     details=self.detailed), ret)
        # 3D array; results in an exception.
        d = [[[1, 2, 3], [4, 5, 6]]]
        self.assertRaises(mb.MiniballValueError, mb.compute, d)
        self.assertRaises(mb.MiniballValueError, mb.compute, np.array(d))
        # String; results in an exception.
        d = "abc"
        self.assertRaises(mb.MiniballTypeError, mb.compute, d)
        self.assertRaises(mb.MiniballValueError, mb.compute, np.array(d))
        # Complex; results in an exception.
        d = [[1+2j, 3+4j], [5+6j, 7+8j]]
        self.assertRaises(mb.MiniballValueError, mb.compute, d)
        self.assertRaises(mb.MiniballValueError, mb.compute, np.array(d))
        # [None]; results in an exception.
        d = [None]
        self.assertRaises(mb.MiniballValueError, mb.compute, d)
        self.assertRaises(mb.MiniballValueError, mb.compute, np.array(d))
        # Mixed container; results in an exception.
        d = [set(), 1, "abc"]
        self.assertRaises(mb.MiniballValueError, mb.compute, d)
        self.assertRaises(mb.MiniballValueError, mb.compute, np.array(d))
        # Mixed container; results in an exception.
        d = [2, 1, None]
        self.assertRaises(mb.MiniballValueError, mb.compute, d)
        self.assertRaises(mb.MiniballValueError, mb.compute, np.array(d))
        # All nans.
        d = [[np.nan, np.nan], [1, np.nan], [0, np.nan]]
        ret = (None, np.nan, dict(center=None, support=None, is_valid=False))
        self.check_ret_cc(mb.compute(d, details=self.detailed), ret)
        self.check_ret_cc(mb.compute(np.array(d),
                                     details=self.detailed), ret)

    def test_corner_cases_max_chord(self):
        # None; results in an exception.
        d = None
        self.assertRaises(mb.MiniballValueError, mb.compute_max_chord, d)
        # Empty array; results in an exception.
        d = []
        self.assertRaises(mb.MiniballValueError, mb.compute_max_chord,
                          d)
        self.assertRaises(mb.MiniballValueError, mb.compute_max_chord,
                          np.array(d, dtype=int))
        self.assertRaises(mb.MiniballValueError, mb.compute_max_chord,
                          np.empty([0, 4], dtype=float))
        # Scalar; is treated as one 1D point [42].
        d = 42
        ret = ([42, 42], 0, dict(center=42, radius=0,
                                 n_support=1, support=[0],
                                 pts_max=[42, 42], ids_max=[0, 0], d_max=0))
        self.check_ret_cc(mb.compute_max_chord(d, details=self.detailed), ret)
        self.check_ret_cc(mb.compute_max_chord(np.array(d, dtype=int),
                                               details=self.detailed), ret)
        # 1D array; is treated as a list of 1D points.
        d = [1, 2, 4, 5]
        ret = ([1, 5], 4, dict(center=3, radius=4, n_support=2,
                               support=[0, 3], pts_max=[1, 5], d_max=4))
        self.check_ret_cc(mb.compute_max_chord(d, details=self.detailed), ret)
        self.check_ret_cc(mb.compute_max_chord(np.array(d, dtype=int),
                                               details=self.detailed), ret)
        # 3D array; results in an exception.
        d = [[[1, 2, 3], [4, 5, 6]]]
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, d)
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, np.array(d))
        # String; results in an exception.
        d = "abc"
        self.assertRaises(mb.MiniballTypeError,
                          mb.compute_max_chord, d)
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, np.array(d))
        # Complex; results in an exception.
        d = [[1+2j, 3+4j], [5+6j, 7+8j]]
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, d)
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, np.array(d))
        # [None]; results in an exception.
        d = [None]
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, d)
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, np.array(d))
        # Mixed container; results in an exception.
        d = [set(), 1, "abc"]
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, d)
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, np.array(d))
        # Mixed container; results in an exception.
        d = [2, 1, None]
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, d)
        self.assertRaises(mb.MiniballValueError,
                          mb.compute_max_chord, np.array(d))


################################################################################
class TestTypesDetailed(TestTypes):
    # Same as TestTypes, but with details.
    detailed = True

    def test_small_ball(self):
        data = [1.0, 1.0001]
        _, _, det = mb.compute(data, details=True)
        self.assertFalse(det["is_valid"])
        _, _, det = mb.compute(data, details=True, tol=1e-10)
        self.assertTrue(det["is_valid"])


################################################################################
class TestRandom(unittest.TestCase):
    dtypes = [np.float32, np.float64, np.float128]
    n_points = [1, 2, 3, 4, 10, 10000]
    n_dims = [0, 1, 2, 3, 4]
    n_reps = 10
    tol = 1e-5

    def setUp(self):
        self.rs = np.random.RandomState(42)

    def test_plain(self):
        for dt in self.dtypes:
            for d in self.n_dims:
                for n in self.n_points:
                    with self.subTest(n=n, d=d, dt=dt):
                        for _ in range(self.n_reps):
                            size = (n, d) if d else (n,)
                            data = self.rs.normal(0, 1, size)
                            data = data.astype(dt)
                            # Compute with and without details.
                            C_A, r2_A = mb.compute(data, details=False)
                            C_B, r2_B, info = mb.compute(data, details=True)
                            np.testing.assert_array_equal(C_A, C_B)
                            self.assertEqual(r2_A, r2_B)
                            if info is None:
                                continue
                            # Related to compute_max_chord().
                            self.assertNotIn("ids_max", info)
                            self.assertNotIn("d_max", info)
                            mb.compute_max_chord(data, info=info)
                            self.assertIn("ids_max", info)
                            self.assertIn("d_max", info)
                            # Upper bound: d_max<=2*radius
                            upper = 2*np.sqrt(r2_A)+self.tol
                            self.assertLessEqual(info["d_max"], upper)

    def test_get_bounding_ball(self):
        """
        See miniball._compat.get_bounding_ball()
        """
        for dt in self.dtypes:
            with self.subTest(dt=dt):
                data = self.rs.normal(0, 1, (100, 5))
                data = data.astype(dt)
                C_A, r2_A = mb.compute(data)
                C_B, r2_B = mb.get_bounding_ball(data)
                np.testing.assert_array_equal(C_A, C_B)
                self.assertEqual(r2_A, r2_B)

    def test_miniballcpp_interface(self):
        """
        See miniball._compat.Miniball()
        """
        for dt in self.dtypes:
            with self.subTest(dt=dt):
                data = self.rs.normal(0, 1, (100, 5))
                data = data.astype(dt)
                C_A, r2_A, info = mb.compute(data, details=True)
                M = mb.Miniball(data)
                C_B = M.center()
                r2_B = M.squared_radius()
                np.testing.assert_array_equal(C_A, C_B)
                self.assertEqual(r2_A, r2_B)
                self.assertEqual(info["relative_error"], M.relative_error())
                self.assertEqual(info["is_valid"], M.is_valid())
                np.testing.assert_almost_equal(info["elapsed"], M.get_time(), 3)


################################################################################
if __name__ == "__main__":
    unittest.main(verbosity=2)
