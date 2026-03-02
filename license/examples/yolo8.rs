use anyhow::{Context, Result};
use onnx_rs::yolo::Yolo;
use std::env;
use std::fs;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <model_path> <images_dir>", args[0]);
        return Ok(());
    }

    let model_path = &args[1];
    let images_dir = &args[2];

    // Initialize YOLO
    // Assuming 640x640, 138 classes as per previous cig context.
    // You might want to make these arguments if generic benchmarking is needed.
    println!("model={}", model_path);
    let yolo = Yolo::new(model_path, 640, 138, 0.25, 0.45).context("Failed to load YOLO model")?;

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
    println!("{} images from {}", image_paths.len(), images_dir);

    let mut total_duration = std::time::Duration::new(0, 0);
    let mut count = 0;

    println!("{:<30} | {:<10} | {:<10}", "Image", "Objects", "Time(ms)");
    println!("{:-<30}-|-{:-<10}-|-{:-<10}", "", "", "");

    for path in image_paths {
        let img_name = path.file_name().unwrap_or_default().to_string_lossy();

        let start_load = Instant::now();
        let img = match image::open(&path) {
            Ok(i) => i,
            Err(e) => {
                println!("Failed to load {}: {}", img_name, e);
                continue;
            }
        };
        // We measure mainly inference time + post-process (which yolo.run does)
        // Loading time is excluded from FPS calculation usually, but let's see user intent.
        // Usually benchmarking focuses on the model run time.

        // Warmup? usually benchmarks do a warmup run, but let's just run simple loop.

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
    }

    if count > 0 {
        let avg_ms = (total_duration.as_secs_f64() * 1000.0) / count as f64;
        let fps = 1000.0 / avg_ms;
        println!("\nBenchmark Results:");
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
