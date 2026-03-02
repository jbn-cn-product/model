use anyhow::{Context, Result};
use image::Rgba;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use onnx_rs::yolo_e2e::YoloEnd2End;
use std::env;
use std::fs;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <model_path> <images_dir> [output_dir]", args[0]);
        return Ok(());
    }

    let model_path = &args[1];
    let images_dir = &args[2];
    let output_dir = if args.len() > 3 { Some(&args[3]) } else { None };

    println!("Loading E2E model from {}", model_path);
    // Note: No num_classes or iou_thres needed for E2E
    let yolo = YoloEnd2End::new(model_path, 640, 0.45).context("Failed to load YOLO E2E model")?;

    let dir_entries =
        fs::read_dir(images_dir).context(format!("Failed to read directory {}", images_dir))?;

    let mut image_paths = Vec::new();
    for entry in dir_entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png" || ext_str == "bmp" {
                    image_paths.push(path);
                }
            }
        }
    }

    if image_paths.is_empty() {
        println!("No images found in {}", images_dir);
        return Ok(());
    }

    image_paths.sort();
    println!("Found {} images in {}", image_paths.len(), images_dir);

    // If output dir specified, create it
    if let Some(out_dir) = output_dir {
        fs::create_dir_all(out_dir).context(format!("Failed to create output dir {}", out_dir))?;
        println!("Saving detailed results to {}", out_dir);
    }

    let mut total_duration = std::time::Duration::new(0, 0);
    let mut count = 0;

    println!("{:<30} | {:<10} | {:<10}", "Image", "Objects", "Time(ms)");
    println!("{:-<30}-|-{:-<10}-|-{:-<10}", "", "", "");

    for path in image_paths {
        let img_name = path.file_name().unwrap_or_default().to_string_lossy();

        let img = match image::open(&path) {
            Ok(i) => i,
            Err(e) => {
                println!("Failed to load {}: {}", img_name, e);
                continue;
            }
        };

        let start_infer = Instant::now();
        let detections = yolo.run(&img)?;
        let duration = start_infer.elapsed();

        total_duration += duration;
        count += 1;

        println!(
            "{:<30} | {:<10} | {:<10.2}",
            img_name,
            detections.len(),
            duration.as_secs_f64() * 1000.0
        );
        let mut objects_info: Vec<(i32, f32)> = detections
            .iter()
            .map(|d| (d.class_id as i32, d.score))
            .collect();
        objects_info.sort_by_key(|k| k.0);
        println!("  Classes: {:?}", objects_info);

        // Visual Verification: Save output image ONLY if output_dir is set
        if let Some(out_dir) = output_dir {
            // Note: drawing and saving is NOT included in `total_duration` since we stopped the timer above
            let mut out_img = img.to_rgba8();
            let color = Rgba([0, 255, 0, 255]); // Green

            for det in &detections {
                let x = det.bbox[0] as i32;
                let y = det.bbox[1] as i32;
                let w = (det.bbox[2] - det.bbox[0]) as u32;
                let h = (det.bbox[3] - det.bbox[1]) as u32;

                draw_hollow_rect_mut(&mut out_img, Rect::at(x, y).of_size(w, h), color);
            }

            let out_path = format!("{}/out_{}", out_dir, img_name);
            use image::DynamicImage;
            let final_img = DynamicImage::ImageRgba8(out_img).to_rgb8();
            if let Err(e) = final_img.save(&out_path) {
                eprintln!("Failed to save output image {}: {}", out_path, e);
            }
        }
    }

    if count > 0 {
        let avg_ms = (total_duration.as_secs_f64() * 1000.0) / count as f64;
        let fps = 1000.0 / avg_ms;
        println!("\nBenchmark E2E Results:");
        println!("  Total Images: {}", count);
        println!(
            "  Total Time:   {:.2} ms",
            total_duration.as_secs_f64() * 1000.0
        );
        println!("  Avg Time:     {:.2} ms/img", avg_ms);
        println!("  FPS:          {:.2}", fps);
    }

    Ok(())
}
