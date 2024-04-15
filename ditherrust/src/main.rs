use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::env;
use std::path::Path;

use image::{ImageBuffer, Luma, open, Pixel, Rgb32FImage};
use image::imageops::FilterType;
use ndarray::{arr2, Array, Array2, ArrayBase, Ix2, OwnedRepr, ShapeBuilder};
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
    println!("{:?} {:?}", raw_img.height(), raw_img.width());

    let img = raw_img.resize(100, 100, FilterType::Triangle).to_rgb32f();
    println!("{:?} {:?}", img.height(), img.width());

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

impl Eq for ErrorPixel {}

impl Ord for ErrorPixel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Less)
    }
}


fn dither_img(img: &ImageBuffer<Luma<f32>, Vec<f32>>)
              -> ImageBuffer<Luma<u8>, Vec<u8>>
{
    let kernel = arr2(&[[0.00354341, 0.01588048, 0.02618249, 0.01588048, 0.00354341],
        [0.01588048, 0.07117138, 0.11734176, 0.07117138, 0.01588048],
        [0.02618249, 0.11734176, 0., 0.11734176, 0.02618249],
        [0.01588048, 0.07117138, 0.11734176, 0.07117138, 0.01588048],
        [0.00354341, 0.01588048, 0.02618249, 0.01588048, 0.00354341]]);

    let shape = (img.width() as usize, img.height() as usize).f();
    let mut already_included = Array2::from_elem(shape, false);
    let mut out = Array2::from_elem(shape, u8::MAX / 2);
    let mut err_img: ArrayBase<OwnedRepr<f32>, Ix2> = Array2::from_shape_vec(shape, img.to_vec()).unwrap() - Array2::from_elem(shape, 50.0);
    let mut errorheap = create_pixel_queue(&err_img);

    while !errorheap.is_empty() {
        let errorpixel = errorheap.pop().unwrap();
        let xy = [errorpixel.x, errorpixel.y];
        if already_included[xy] || (err_img[xy].abs() != errorpixel.error) {
            continue;
        }
        already_included[xy] = true;
        out[xy] = match errorpixel.sign {
            Sign::Positive => { u8::MAX }
            Sign::Negative => { 0 }
        };
        let error = &kernel * (50.0 - errorpixel.error) * match errorpixel.sign {
            Sign::Positive => { -1.0 }
            Sign::Negative => { 1.0 }
        };
        add_error(&mut err_img, error, xy);
        errorheap = create_pixel_queue(&err_img);  // TODO: only do the pixels we've changed
    }
    let out_img = ImageBuffer::from_fn(img.width(), img.height(), |x, y| {
        Luma([out[[x as usize, y as usize]]])
    });
    out_img
}

fn add_error(err_img: &mut ArrayBase<OwnedRepr<f32>, Ix2>, error: Array<f32, Ix2>, centre: [usize; 2]) {
    let [error_width, error_height] = error.shape() else { unreachable!() };
    let [centre_x, centre_y] = centre;
    for i in 0..error_width.clone() {
        for j in 0..error_height.clone() {
            let x;
            let y;
            match (centre_x + i).checked_sub(error_width / 2) {
                None => { continue; }
                Some(val) => { x = val }
            }
            match (centre_y + j).checked_sub(error_height / 2) {
                None => { continue; }
                Some(val) => { y = val }
            }
            if (x + 1 > err_img.shape()[0]) || (y + 1 > err_img.shape()[1]) {
                continue;
            }
            err_img[[x, y]] += error[[i, j]];
        }
    }
}

fn create_pixel_queue(err_img: &ArrayBase<OwnedRepr<f32>, Ix2>) -> BinaryHeap<ErrorPixel> {
    let x1: Vec<ErrorPixel> = err_img.indexed_iter().map(|((x, y), error)| ErrorPixel {
        error: error.abs(),
        sign: if error.clone() < 0.0 { Sign::Negative } else { Sign::Positive },
        x,
        y,
    }).collect();
    let errorheap: BinaryHeap<ErrorPixel> = BinaryHeap::from(x1);
    errorheap
}




