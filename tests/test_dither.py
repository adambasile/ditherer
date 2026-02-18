from pathlib import Path

import numpy as np
import pytest
from skimage import io

from baddither import dither

EXAMPLE_IMG_DIR = Path(__file__).parent.parent / "example_images"


@pytest.mark.parametrize(
    ("size", "expected_file"),
    [
        (100, "100px.png"),
        (200, "200px.png"),
        (300, "300px.png"),
    ],
)
def test_dither_variants(size, expected_file):
    """Tests dithering at various resolutions."""
    input_path = EXAMPLE_IMG_DIR / "circe.jpg"
    expected_path = EXAMPLE_IMG_DIR / expected_file

    dithered = dither.dither(dither.read(input_path, size))
    expected = io.imread(expected_path)

    np.testing.assert_equal(dithered, expected)
