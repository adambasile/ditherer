use std::env;
use std::path::Path;
use image::{ImageBuffer, Luma, open, Pixel, Rgb32FImage};
use palette::{IntoColor, Lab, Srgb};
use palette::white_point::D65;

fn main() {
    let args: Vec<String> = env::args().collect();
    let infile = Path::new(&args[1]);
    let outfile = Path::new(&args[2]);

    println!("Reading {:?}", infile);
    let img = open(infile).unwrap().to_rgb32f();

    let out_img = convert_img_to_luma(img);
    println!("Writing {:?}", outfile);
    out_img.save(outfile).unwrap();
}

fn convert_img_to_luma(img: Rgb32FImage) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let out_img = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let pixel = img.get_pixel(x, y).to_rgb();
        let raw: Srgb<f32> = Srgb::new(pixel.channels()[0], pixel.channels()[1], pixel.channels()[2]);

        let lab: Lab<D65, f32> = raw.into_color();
        let luma: Luma<u8> = Luma([((lab.l) * 255.0 / 100.0).round() as u8]);
        luma
    });
    out_img
}

fn dither_img(img: ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let _kernel = ([[0.00354341, 0.01588048, 0.02618249, 0.01588048, 0.00354341],
        [0.01588048, 0.07117138, 0.11734176, 0.07117138, 0.01588048],
        [0.02618249, 0.11734176, 0., 0.11734176, 0.02618249],
        [0.01588048, 0.07117138, 0.11734176, 0.07117138, 0.01588048],
        [0.00354341, 0.01588048, 0.02618249, 0.01588048, 0.00354341]]);
    img.clone()
}


