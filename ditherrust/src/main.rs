use std::collections::BinaryHeap;
use std::env;
use std::path::Path;

use image::{ImageBuffer, Luma, open, Pixel, Rgb32FImage};
use ndarray::Dimension;
use ndarray::prelude::*;
use palette::{IntoColor, Lab, Srgb};
use palette::white_point::D65;

fn main() {
    use std::time::Instant;
    let now = Instant::now();

    let args: Vec<String> = env::args().collect();
    let infile = Path::new(&args[1]);
    let outfile = Path::new(&args[2]);

    println!("Reading {:?}", infile);
    let raw_img = open(infile).unwrap();

    let img = raw_img.to_rgb32f();

    let luminance = convert_img_to_relative_luminance(&img);

    let dithered = dither_img(&luminance);

    println!("Writing {:?}", outfile);
    let out_img = convert_luminance_to_int(&luminance);
    dithered.save(outfile).unwrap();
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);



}

fn convert_luminance_to_int(img: &ImageBuffer<Luma<f32>, Vec<f32>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let pixel = img.get_pixel(x, y);
        let l: Luma<u8> = Luma([(pixel.channels()[0] * u8::MAX as f32 / 100.0).round() as u8]);
        l
    })
}

fn convert_img_to_relative_luminance(img: &Rgb32FImage) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        let pixel = img.get_pixel(x, y).to_rgb();
        let raw: Srgb<f32> = Srgb::new(pixel.channels()[0], pixel.channels()[1], pixel.channels()[2]);
        let lab: Lab<D65, f32> = raw.into_color();
        let luma: Luma<f32> = Luma([lab.l]);
        luma
    })
}



#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Sign {
    Positive,
    Negative,
}

#[derive(Debug, PartialEq, PartialOrd)]
struct ErrorPixel {
    error: f32,
    sign: Sign,
    x: usize,
    y: usize,
}

fn dither_img(img: &ImageBuffer<Luma<f32>, Vec<f32>>)
              -> ImageBuffer<Luma<u8>, Vec<u8>>
{
    let _kernel = arr2(&[[0.00354341, 0.01588048, 0.02618249, 0.01588048, 0.00354341],
        [0.01588048, 0.07117138, 0.11734176, 0.07117138, 0.01588048],
        [0.02618249, 0.11734176, 0., 0.11734176, 0.02618249],
        [0.01588048, 0.07117138, 0.11734176, 0.07117138, 0.01588048],
        [0.00354341, 0.01588048, 0.02618249, 0.01588048, 0.00354341]]);

    let shape = (img.width() as usize, img.height() as usize).f();
    let already_included = Array2::from_elem(shape, false);
    let out = Array2::from_elem(shape, u8::MAX / 2);
    let err_img = Array2::from_shape_vec(shape, img.to_vec()).unwrap() - Array2::from_elem(shape, 0.5);
    let out_img = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        Luma([out[[x as usize, y as usize]]])
    });
    out_img
}




