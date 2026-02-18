import argparse

from baddither.dither import process


def main() -> None:
    parser = argparse.ArgumentParser(description="Dither an image.")
    parser.add_argument("input", type=str, help="path of image to dither")
    parser.add_argument("output", type=str, help="output path")
    parser.add_argument(
        "-s", "--size", type=int, help="Proportionally resize largest dimension of image to this value before dithering"
    )
    args = parser.parse_args()
    process(args.input, args.size, args.output)
