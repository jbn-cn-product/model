use ab_glyph::{FontRef, PxScale};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use rknn_rs::Object;

const FONT_PATH: &str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";
pub const UNKNOWN_CLASS_ID: i32 = 124;

fn draw_bbox(mut image: RgbImage, detections: &[Object], output_path: &str) -> Result<(), String> {
    let font_data = std::fs::read(FONT_PATH)
        .map_err(|e| format!("Failed to load font from {}: {}", FONT_PATH, e))?;
    let font = FontRef::try_from_slice(&font_data)
        .map_err(|e| format!("Failed to parse font from {}: {:?}", FONT_PATH, e))?;
    let scale = PxScale { x: 20.0, y: 20.0 };
    let white = Rgb([255u8, 255u8, 255u8]);
    // let black = Rgb([0u8, 0u8, 0u8]); // Not used

    for det in detections {
        let x1 = det.bbox[0] as i32;
        let y1 = det.bbox[1] as i32;
        let x2 = det.bbox[2] as i32;
        let y2 = det.bbox[3] as i32;

        let width = (x2 - x1).max(1) as u32;
        let height = (y2 - y1).max(1) as u32;
        let rect = Rect::at(x1, y1).of_size(width, height);

        let color = if det.class_id == UNKNOWN_CLASS_ID {
            Rgb([255u8, 0u8, 0u8]) // Red for unknown
        } else {
            Rgb([0u8, 255u8, 0u8]) // Green for known classes
        };

        draw_hollow_rect_mut(&mut image, rect, color);

        let class_label = if det.class_id == UNKNOWN_CLASS_ID {
            "Unknown".to_string()
        } else {
            format!("Class {} ({:.2})", det.class_id, det.confidence)
        };

        draw_text_mut(
            &mut image,
            white,
            x1,
            y1 - 20, // Position above the box
            scale,
            &font,
            &class_label,
        );
    }

    image
        .save(output_path)
        .map_err(|e| format!("Failed to save output image to {}: {}", output_path, e))?;

    println!("annotated-image={}", output_path);
    Ok(())
}
