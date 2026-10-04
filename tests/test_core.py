import unittest
import numpy as np
from lufa.retrieval import nearest, resample


class RetrievalTests(unittest.TestCase):
    def test_cosine_is_norm_invariant(self):
        indices, scores = nearest(np.array([2., 0.]), np.array([[1., 1.], [100., 0.], [-1., 0.]]), 2)
        self.assertEqual(indices.tolist(), [1, 0])
        self.assertAlmostEqual(float(scores[0]), 1.)

    def test_resampling_preserves_endpoints(self):
        clip = np.stack([np.zeros(9), np.ones(9)])
        result = resample(clip, 5)
        np.testing.assert_allclose(result[0], clip[0])
        np.testing.assert_allclose(result[-1], clip[-1])
        np.testing.assert_allclose(result[2], .5)

    def test_degenerate_queries_rejected(self):
        with self.assertRaises(ValueError):
            nearest(np.zeros(2), np.ones((2, 2)))


if __name__ == "__main__":
    unittest.main()
