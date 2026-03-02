use image::Rgba;
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use rknn_rs::Context;
use rknn_rs::yolo::Yolo;
use std::env;
use std::fs;
use std::time::{Duration, Instant};

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <model_path> <images_dir> [output_dir]", args[0]);
        return Ok(());
    }

    let model_path = &args[1];
    let images_dir = &args[2];
    let output_dir = if args.len() > 3 { Some(&args[3]) } else { None };

    println!("Loading RKNN model from {}", model_path);
    // Core mask 0 (auto), Conf 0.25
    let ctx = Yolo::new(0, model_path, 0.25)?;

    let dir_paths = fs::read_dir(images_dir)
        .map_err(|e| format!("Failed to read directory {}: {}", images_dir, e))?;

    let mut image_paths = Vec::new();
    for entry in dir_paths {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ["jpg", "jpeg", "png", "bmp"].contains(&ext_str.as_str()) {
                        image_paths.push(path);
                    }
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

    // Create output dir if needed
    if let Some(out_dir) = output_dir {
        fs::create_dir_all(out_dir)
            .map_err(|e| format!("Failed create output dir {}: {}", out_dir, e))?;
        println!("Saving detailed results to {}", out_dir);
    }

    let mut total_infer_time = Duration::new(0, 0);
    let mut count = 0;

    println!(
        "{:<30} | {:<7} | {:<35}",
        "Image", "Objects", "Timing (Pre/Inf/Post/Total)"
    );
    println!("{:-<30}-|-{:-<7}-|-{:-<35}", "", "", "");

    for path in image_paths {
        let img_name = path.file_name().unwrap_or_default().to_string_lossy();

        let image_bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                println!("Failed to read {}: {}", img_name, e);
                continue;
            }
        };

        // For drawing later (load before bench loop to avoid IO/Decode cost in timing if possible,
        // but pre() also decodes. Let's load purely for drawing if output is requested)
        let original_img = if output_dir.is_some() {
            match image::load_from_memory(&image_bytes) {
                Ok(i) => Some(i),
                Err(_) => None,
            }
        } else {
            None
        };

        // 1. Pre-process (includes decode)
        // We usually count this in E2E pipeline benchmarks unless measuring PURE NPU inference.
        // But let's measure mostly Inference + Post, as Pre involves CPU decoding which might vary.
        // Actually, let's call pre() outside the timer if we want to measure Model Performance.
        // If we want "System Performance", we include everything.
        // Let's include Pre because pre() does Letterboxing which is part of latency.

        let start_total = Instant::now();

        let t0 = Instant::now();
        let (pre_data, letterbox_info) = ctx.pre(&image_bytes)?;
        let t_pre = t0.elapsed();

        let t1 = Instant::now();
        let raw_outputs = ctx.inference(&pre_data)?;
        let t_infer = t1.elapsed();

        let t2 = Instant::now();
        let objects = ctx.process(&raw_outputs, &letterbox_info)?;
        let t_post = t2.elapsed();

        let duration = start_total.elapsed();

        total_infer_time += duration;
        count += 1;

        println!(
            "{:<30} | {:<7} | Pre:{:.1}ms Inf:{:.1}ms Post:{:.1}ms | Total:{:.1}ms",
            img_name,
            objects.len(),
            t_pre.as_secs_f64() * 1000.0,
            t_infer.as_secs_f64() * 1000.0,
            t_post.as_secs_f64() * 1000.0,
            duration.as_secs_f64() * 1000.0
        );
        let mut objects_info: Vec<(i32, f32)> =
            objects.iter().map(|o| (o.class_id, o.confidence)).collect();
        objects_info.sort_by_key(|k| k.0);
        println!("  Classes: {:?}", objects_info);

        // Verification Drawing (Not timed)
        if let Some(out_dir) = output_dir {
            if let Some(img) = original_img {
                let mut out_img = img.to_rgba8();
                let color = Rgba([0, 255, 0, 255]); // Green

                for obj in &objects {
                    let x = obj.bbox[0] as i32;
                    let y = obj.bbox[1] as i32;
                    let w = (obj.bbox[2] - obj.bbox[0]) as u32;
                    let h = (obj.bbox[3] - obj.bbox[1]) as u32;
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
    }

    if count > 0 {
        let avg_ms = (total_infer_time.as_secs_f64() * 1000.0) / count as f64;
        let fps = 1000.0 / avg_ms;
        println!("\nBenchmark Analysis Results:");
        println!("  Total Images: {}", count);
        println!(
            "  Total Time:   {:.2} ms",
            total_infer_time.as_secs_f64() * 1000.0
        );
        println!("  Avg Time:     {:.2} ms/img", avg_ms);
        println!("  FPS:          {:.2}", fps);
    }

    Ok(())
}
