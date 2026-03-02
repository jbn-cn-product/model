use anyhow::Result;
use image::{DynamicImage, GenericImageView, Pixel, imageops::FilterType};
use ndarray::Array4;
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: usize,
}

pub struct Yolo {
    session: Session,
    input_size: u32,
    num_classes: usize,
    conf_thres: f32,
    iou_thres: f32,
}

impl Yolo {
    pub fn new(
        model_path: &str,
        input_size: u32,
        num_classes: usize,
        conf_thres: f32,
        iou_thres: f32,
    ) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session,
            input_size,
            num_classes,
            conf_thres,
            iou_thres,
        })
    }

    pub fn run(&self, img: &DynamicImage) -> Result<Vec<Detection>> {
        let input_size = self.input_size;
        let (w, h) = img.dimensions();
        let scale = (input_size as f32 / w as f32).min(input_size as f32 / h as f32);
        let new_w = (w as f32 * scale) as u32;
        let new_h = (h as f32 * scale) as u32;

        let resized = img.resize_exact(new_w, new_h, FilterType::Triangle);

        // Fill with 114/255 (Gray)
        let mut input_tensor = Array4::<f32>::from_elem(
            (1, 3, input_size as usize, input_size as usize),
            114.0 / 255.0,
        );

        // Center Padding
        let pad_w = (input_size - new_w) / 2;
        let pad_h = (input_size - new_h) / 2;

        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = resized.get_pixel(x, y);
                let rgb = pixel.to_rgb();

                let dest_x = (x + pad_w) as usize;
                let dest_y = (y + pad_h) as usize;

                input_tensor[[0, 0, dest_y, dest_x]] = rgb[0] as f32 / 255.0;
                input_tensor[[0, 1, dest_y, dest_x]] = rgb[1] as f32 / 255.0;
                input_tensor[[0, 2, dest_y, dest_x]] = rgb[2] as f32 / 255.0;
            }
        }

        // Pass (shape, slice) tuple to Value::from_array
        let shape = input_tensor.shape();
        let slice = input_tensor
            .as_slice()
            .ok_or(anyhow::anyhow!("Input tensor is not contiguous"))?;
        let input_value = Value::from_array((shape, slice))?;
        let inputs = inputs!["images" => input_value]?;
        let outputs = self.session.run(inputs)?;
        let tensor = outputs["output0"].try_extract_tensor::<f32>()?;
        let shape = tensor.shape();
        let data = tensor
            .as_slice()
            .ok_or(anyhow::anyhow!("Output tensor is not contiguous"))?;

        // Layout is [1, 5, 8400] -> Planar
        let mut detections: Vec<Detection> = Vec::new();
        let num_anchors = shape[2] as usize;
        let stride = num_anchors;

        for i in 0..num_anchors {
            // Find max score among classes
            let mut max_score = -1.0;
            let mut class_id = 0;
            for c in 0..self.num_classes {
                let score = data[i + (4 + c) * stride];
                if score > max_score {
                    max_score = score;
                    class_id = c;
                }
            }

            if max_score > self.conf_thres {
                let x = data[i];
                let y = data[i + stride];
                let w = data[i + 2 * stride];
                let h = data[i + 3 * stride];

                // Adjust for padding
                let x = x - pad_w as f32;
                let y = y - pad_h as f32;

                let x1 = (x - w / 2.0) / scale;
                let y1 = (y - h / 2.0) / scale;
                let x2 = (x + w / 2.0) / scale;
                let y2 = (y + h / 2.0) / scale;
                detections.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    score: max_score,
                    class_id,
                });
            }
        }

        nms(&mut detections, self.iou_thres);
        Ok(detections)
    }
}

fn nms(detections: &mut Vec<Detection>, iou_thres: f32) {
    detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    let mut keep = Vec::new();
    let mut active = vec![true; detections.len()];
    for i in 0..detections.len() {
        if !active[i] {
            continue;
        }
        keep.push(i);
        for j in (i + 1)..detections.len() {
            if !active[j] {
                continue;
            }
            let iou = calculate_iou(&detections[i], &detections[j]);
            if iou > iou_thres {
                active[j] = false;
            }
        }
    }
    let kept_detections: Vec<Detection> = keep.into_iter().map(|i| detections[i]).collect();
    *detections = kept_detections;
}

fn calculate_iou(a: &Detection, b: &Detection) -> f32 {
    let x1 = a.bbox[0].max(b.bbox[0]);
    let y1 = a.bbox[1].max(b.bbox[1]);
    let x2 = a.bbox[2].min(b.bbox[2]);
    let y2 = a.bbox[3].min(b.bbox[3]);
    if x2 < x1 || y2 < y1 {
        return 0.0;
    }
    let intersection = (x2 - x1) * (y2 - y1);
    let area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1]);
    let area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1]);
    intersection / (area_a + area_b - intersection)
}
