from typing import Annotated

import typer
from skimage import io

from baddither.dither import dither, read

app = typer.Typer()


@app.command()
def main(
    input_path: Annotated[str, typer.Argument(help="Path of image to dither")],
    output_path: Annotated[str, typer.Argument(help="Output path")],
    size: Annotated[
        int | None,
        typer.Option("--size", "-s", help="Proportionally resize largest dimension of image to this value"),
    ] = None,
) -> None:
    """
    Dither an image.
    """
    img = read(input_path, size)
    dithered = dither(img)
    io.imsave(output_path, dithered)
