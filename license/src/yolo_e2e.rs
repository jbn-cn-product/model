use anyhow::Result;
use image::{DynamicImage, GenericImageView, Pixel, imageops::FilterType};
use ndarray::Array4;
use ort::{
    inputs,
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};

#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub bbox: [f32; 4],
    pub score: f32,
    pub class_id: usize,
}

pub struct YoloEnd2End {
    session: Session,
    input_size: u32,
    conf_thres: f32,
}

impl YoloEnd2End {
    pub fn new(model_path: &str, input_size: u32, conf_thres: f32) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self {
            session,
            input_size,
            conf_thres,
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
        let shape = tensor.shape(); // [1, N, 6]
        let data = tensor
            .as_slice()
            .ok_or(anyhow::anyhow!("Output tensor is not contiguous"))?;

        let mut detections = Vec::new();

        // Ensure shape is [1, N, 6]
        if shape.len() != 3 || shape[2] != 6 {
            return Err(anyhow::anyhow!(
                "Expected output shape [1, N, 6], got {:?}",
                shape
            ));
        }

        let num_dets = shape[1];

        for i in 0..num_dets {
            let offset = i * 6;
            let score = data[offset + 4];
            if score > self.conf_thres {
                let class_id = data[offset + 5] as usize;
                let x1_in = data[offset + 0];
                let y1_in = data[offset + 1];
                let x2_in = data[offset + 2];
                let y2_in = data[offset + 3];

                // Undo padding/scaling
                let x1 = (x1_in - pad_w as f32) / scale;
                let y1 = (y1_in - pad_h as f32) / scale;
                let x2 = (x2_in - pad_w as f32) / scale;
                let y2 = (y2_in - pad_h as f32) / scale;

                detections.push(Detection {
                    bbox: [x1, y1, x2, y2],
                    score,
                    class_id,
                });
            }
        }

        Ok(detections)
    }
}
