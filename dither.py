import numpy as np
from skimage import io, color, transform, filters


def dither(img):
    current = np.zeros(np.array(img.shape) + 4)
    current[:] = np.nan
    current[2:-2, 2:-2] = img - 0.5

    dithered = np.full_like(current, fill_value=np.nan)

    kernel = np.zeros([5, 5])
    kernel[2, 2] = 1
    kernel = filters.gaussian(kernel)
    kernel[2, 2] = 0
    kernel = kernel / kernel.sum()

    for _ in range(img.size):
        x, y = np.unravel_index(np.nanargmax(np.abs(current)), current.shape)
        val = current[x, y]
        current[x, y] = np.nan
        dithered[x, y] = 0.5 + np.sign(val) / 2
        error = val - np.sign(val) / 2
        current[x - 2 : x + 3, y - 2 : y + 3] += error * kernel
    return dithered[2:-2, 2:-2]


def read(path, size=None):
    raw_img = color.rgb2lab(io.imread(path)[:, :, :3])[:, :, 0] / 100
    if size is None:
        return raw_img
    else:
        return transform.resize(raw_img, (size * np.array(raw_img.shape[:2]) / max(raw_img.shape[:2])).round())


def process(path, size, output_path):
    img = read(path, size)
    dithered = dither(img)
    io.imsave(output_path, 255 * dithered.astype("uint8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dither an image.")
    parser.add_argument("input", type=str, help="path of image to dither")
    parser.add_argument("output", type=str, help="output path")
    parser.add_argument(
        "-s", "--size", type=int, help="Proportionally resize largest dimension of image to this value before dithering"
    )
    args = parser.parse_args()
    process(args.input, args.size, args.output)
