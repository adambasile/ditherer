import unittest
from pathlib import Path

import numpy as np
from skimage import io

from baddither import dither

EXAMPLE_IMG_DIR = Path(__file__).parent.parent / "example_images"


class TestDither(unittest.TestCase):
    def test_dither100(self):
        dithered = dither.dither(dither.read(EXAMPLE_IMG_DIR / "circe.jpg", 100))
        expected = io.imread(EXAMPLE_IMG_DIR / "100px.png")
        np.testing.assert_equal(dithered, expected)

    def test_dither200(self):
        dithered = dither.dither(dither.read(EXAMPLE_IMG_DIR / "circe.jpg", 200))
        expected = io.imread(EXAMPLE_IMG_DIR / "200px.png")
        np.testing.assert_equal(dithered, expected)

    def test_dither300(self):
        dithered = dither.dither(dither.read(EXAMPLE_IMG_DIR / "circe.jpg", 300))
        expected = io.imread(EXAMPLE_IMG_DIR / "300px.png")
        np.testing.assert_equal(dithered, expected)
