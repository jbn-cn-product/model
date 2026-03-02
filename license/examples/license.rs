use anyhow::Result;
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgb};
use onnx_rs::pp::{Det, Rec};
use onnx_rs::yolo::Yolo;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 6 {
        println!(
            "Usage: {} <yolo> <pp_det> <pp_rec> <pp_rec_vocab> <image>",
            args[0]
        );
        return Ok(());
    }

    let yolo = Yolo::new(&args[1], 640, 1, 0.25, 0.45).expect("Failed to load yolo");
    let pp_det = Det::new(&args[2]).expect("Failed to load pp det");
    let pp_rec = Rec::new(&args[3], &args[4]).expect("Failed to load pp rec");
    let img = image::open(&args[5]).expect("Failed to open image");

    let mut ocr_input_img = img;
    let detections = yolo.run(&ocr_input_img)?;
    if let Some(det) = detections.first() {
        println!("YOLO Detected: {:?}", det);
        let (w, h) = ocr_input_img.dimensions();
        let x1 = det.bbox[0].max(0.0) as u32;
        let y1 = det.bbox[1].max(0.0) as u32;
        let x2 = det.bbox[2].min(w as f32) as u32;
        let y2 = det.bbox[3].min(h as f32) as u32;
        let width = x2.saturating_sub(x1);
        let height = y2.saturating_sub(y1);

        if width > 0 && height > 0 {
            ocr_input_img = ocr_input_img.crop_imm(x1, y1, width, height);
        } else {
            println!("Invalid YOLO crop, using full image");
        }
    } else {
        println!("YOLO: No detection found, using full image");
    }

    // Pad to square to match chris_ocr.py logic
    let (w, h) = ocr_input_img.dimensions();
    let max_side = w.max(h);
    let mut padded_buf = ImageBuffer::from_pixel(max_side, max_side, Rgb([0, 0, 0]));

    let img_rgb = ocr_input_img.to_rgb8();
    padded_buf
        .copy_from(&img_rgb, 0, 0)
        .expect("Failed to copy image to padded buffer");
    let padded_img = DynamicImage::ImageRgb8(padded_buf);

    let boxes = pp_det.run(&padded_img).expect("Detection failed");
    println!("Found {} objects", boxes.len());

    for (i, bbox) in boxes.iter().enumerate() {
        // Crop
        // Ensure crop is within bounds
        let x = bbox[0].max(0.0) as u32;
        let y = bbox[1].max(0.0) as u32;
        let mut width = (bbox[2] - bbox[0]).max(1.0) as u32;
        let mut height = (bbox[3] - bbox[1]).max(1.0) as u32;

        let (pw, ph) = padded_img.dimensions();
        if x + width > pw {
            width = pw - x;
        }
        if y + height > ph {
            height = ph - y;
        }

        if width == 0 || height == 0 {
            continue;
        }

        let crop = padded_img.crop_imm(x, y, width, height);

        // Run Recognition
        let text = pp_rec.run(&crop).unwrap_or_default();

        println!("{}. {:?} {}", i, bbox, text);
    }

    Ok(())
}
