import heapq
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from skimage import color, filters, io, transform


def create_pixel_queue(img, offset_x, offset_y):
    x, y = np.mgrid[slice(img.shape[0]), slice(img.shape[1])]
    x, y = x.flatten() + offset_x, y.flatten() + offset_y
    flattened = img.flatten()
    out = np.array([-np.abs(flattened), np.sign(flattened), x, y]).transpose()[~np.isnan(flattened)].reshape(-1, 4)
    return [tuple(pixel) for pixel in out.tolist()]


def pop_pixel(pixel_queue, to_ignore):
    pixel = heapq.heappop(pixel_queue)
    while pixel in to_ignore:
        pixel = heapq.heappop(pixel_queue)
        if not pixel_queue:
            return None
    val, sign, x, y = pixel
    return val, sign, int(x), int(y)


def dither(img):
    current = np.zeros(np.array(img.shape) + 4)
    current[:] = np.nan
    current[2:-2, 2:-2] = img - 0.5
    pixel_queue = create_pixel_queue(current, 0, 0)
    heapq.heapify(pixel_queue)

    dithered = np.full_like(current, fill_value=np.nan)
    to_ignore = set()

    kernel = np.zeros([5, 5])
    kernel[2, 2] = 1
    kernel = filters.gaussian(kernel)
    kernel[2, 2] = 0
    kernel = kernel / kernel.sum()

    for _i in range(img.size):
        pixel = pop_pixel(pixel_queue, to_ignore)
        if pixel is None:
            break
        to_ignore.add(pixel)
        negabsval, sign, x, y = pixel
        current[x, y] = np.nan
        dithered[x, y] = 0.5 + sign / 2
        error = -sign * (0.5 + negabsval)

        x_slice = slice(x - 2, x + 3)
        y_slice = slice(y - 2, y + 3)

        deprecated_pixels = create_pixel_queue(current[x_slice, y_slice], x_slice.start, y_slice.start)
        to_ignore.update(deprecated_pixels)

        current[x_slice, y_slice] += error * kernel
        updated_pixels = create_pixel_queue(current[x_slice, y_slice], x_slice.start, y_slice.start)
        to_ignore.difference_update(updated_pixels)
        for pixel in updated_pixels:
            heapq.heappush(pixel_queue, pixel)

    assert not np.isnan(dithered[2:-2, 2:-2]).any()
    return 255 * dithered[2:-2, 2:-2].astype("uint8")


def read(path, size=None):
    raw_img = color.rgb2lab(io.imread(path)[:, :, :3])[:, :, 0] / 100
    if size is None:
        return raw_img
    ratio = Decimal(size) / max(*raw_img.shape[:2])
    return transform.resize(
        raw_img, tuple(int((ratio * i).to_integral_value(ROUND_HALF_UP)) for i in raw_img.shape[:2])
    )
