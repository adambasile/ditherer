import unittest

import numpy as np
from skimage import io

import dither


class TestDither(unittest.TestCase):
    def test_dither100(self):
        dithered = dither.dither(dither.read("example_images/circe.jpg", 100))
        expected = io.imread("example_images/100px.png")
        np.testing.assert_equal(dithered, expected)

    def test_dither200(self):
        dithered = dither.dither(dither.read("example_images/circe.jpg", 200))
        expected = io.imread("example_images/200px.png")
        np.testing.assert_equal(dithered, expected)

    def test_dither300(self):
        dithered = dither.dither(dither.read("example_images/circe.jpg", 300))
        expected = io.imread("example_images/300px.png")
        np.testing.assert_equal(dithered, expected)
