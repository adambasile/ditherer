import unittest

import numpy as np
from skimage import io

import dither


class TestDither(unittest.TestCase):
    def test_dither(self):
        dithered = dither.dither(dither.read("example_images/circe.jpg", 100))
        expected = io.imread("example_images/100px.png")
        np.testing.assert_equal(dithered, expected)
