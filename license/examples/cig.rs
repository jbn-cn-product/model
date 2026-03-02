use anyhow::Result;
use onnx_rs::yolo::Yolo;
use std::env;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("Usage: {} <yolo> <image>", args[0]);
        return Ok(());
    }

    let yolo = Yolo::new(&args[1], 640, 138, 0.20, 0.45)?;
    let img = image::open(&args[2]).expect("Failed to open image");
    let detections = yolo.run(&img)?;

    println!("detections={}", detections.len());
    for det in detections {
        println!("{:?}", det);
    }

    Ok(())
}
