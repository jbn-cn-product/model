use anyhow::Result;
use image::{imageops::FilterType, DynamicImage, GenericImageView, Pixel};
use ndarray::Array4;
use ort::{
    inputs,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::fs;

pub struct Det {
    session: Session,
}

impl Det {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn run(&self, img: &DynamicImage) -> Result<Vec<[f32; 4]>> {
        let (w, h) = img.dimensions();
        let max_dim = w.max(h);
        let mut padded_img = image::RgbImage::new(max_dim, max_dim); // Initialized to 0 (black)

        let img_rgb = img.to_rgb8();
        for y in 0..h {
            for x in 0..w {
                padded_img.put_pixel(x, y, *img_rgb.get_pixel(x, y));
            }
        }

        let target_size = 960;
        let resized =
            image::imageops::resize(&padded_img, target_size, target_size, FilterType::Triangle);

        let new_w = target_size;
        let new_h = target_size;

        let mut input_tensor = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
        // Normalize: (val/255 - mean) / std
        // Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        for y in 0..new_h {
            for x in 0..new_w {
                let pixel = resized.get_pixel(x, y);
                let rgb = pixel.to_rgb();
                // Use BGR directly to match Python
                let bgr = [rgb[2], rgb[1], rgb[0]];
                for c in 0..3 {
                    let val = bgr[c] as f32 / 255.0;
                    let norm = (val - mean[c]) / std[c];
                    input_tensor[[0, c as usize, y as usize, x as usize]] = norm;
                }
            }
        }

        // Pass (shape, slice) tuple to Value::from_array to bypass ndarray version issues
        let shape = input_tensor.shape();
        let slice = input_tensor
            .as_slice()
            .ok_or(anyhow::anyhow!("Input tensor is not contiguous"))?;
        let input_value = Value::from_array((shape, slice))?;
        let inputs = inputs!["x" => input_value]?;
        let outputs = self.session.run(inputs)?;

        let tensor = outputs[0].try_extract_tensor::<f32>()?;
        let shape = tensor.shape();
        let map_h = shape[2] as usize;
        let map_w = shape[3] as usize;
        let data = tensor
            .as_slice()
            .ok_or(anyhow::anyhow!("Output tensor is not contiguous"))?;

        // Simple Post-processing: Threshold + Connected Components -> BBox
        let threshold = 0.2; // Lowered from 0.3
        let mut visited = vec![false; map_h * map_w];
        let mut boxes: Vec<[f32; 4]> = Vec::new();

        for y in 0..map_h {
            for x in 0..map_w {
                let idx = y * map_w + x;
                if data[idx] > threshold && !visited[idx] {
                    // BFS to find component
                    let mut min_x = x;
                    let mut max_x = x;
                    let mut min_y = y;
                    let mut max_y = y;
                    let mut q = Vec::new();
                    q.push((x, y));
                    visited[idx] = true;

                    while let Some((cx, cy)) = q.pop() {
                        if cx < min_x {
                            min_x = cx;
                        }
                        if cx > max_x {
                            max_x = cx;
                        }
                        if cy < min_y {
                            min_y = cy;
                        }
                        if cy > max_y {
                            max_y = cy;
                        }

                        // 4-connectivity
                        let neighbors = [
                            (cx.wrapping_sub(1), cy),
                            (cx + 1, cy),
                            (cx, cy.wrapping_sub(1)),
                            (cx, cy + 1),
                        ];

                        for (nx, ny) in neighbors {
                            if nx < map_w && ny < map_h {
                                let nidx = ny * map_w + nx;
                                if !visited[nidx] && data[nidx] > threshold {
                                    visited[nidx] = true;
                                    q.push((nx, ny));
                                }
                            }
                        }
                    }

                    // Calculate box score (mean of probability map in the box)
                    let mut box_score_sum = 0.0;
                    let mut box_pixel_count = 0;
                    for by in min_y..=max_y {
                        for bx in min_x..=max_x {
                            box_score_sum += data[by * map_w + bx];
                            box_pixel_count += 1;
                        }
                    }
                    #[allow(unused_variables)]
                    let box_score = if box_pixel_count > 0 {
                        box_score_sum / box_pixel_count as f32
                    } else {
                        0.0
                    };

                    // Filter small
                    if (max_x - min_x) < 3 || (max_y - min_y) < 3 {
                        continue;
                    }

                    // Scale back to original image
                    let scale = max_dim as f32 / target_size as f32;

                    // Expand box (unclip approx)
                    let w_box = (max_x - min_x) as f32;
                    let h_box = (max_y - min_y) as f32;
                    let area = w_box * h_box;
                    let perimeter = 2.0 * (w_box + h_box);
                    let unclip_ratio = 1.6; // Increased from 1.5
                    let pad = if perimeter > 0.0 {
                        area * unclip_ratio / perimeter
                    } else {
                        0.0
                    };

                    let x1 = (min_x as f32 - pad).max(0.0) * scale;
                    let y1 = (min_y as f32 - pad).max(0.0) * scale;
                    let x2 = (max_x as f32 + pad).min(map_w as f32) * scale;
                    let y2 = (max_y as f32 + pad).min(map_h as f32) * scale;

                    boxes.push([x1, y1, x2, y2]);
                }
            }
        }

        Ok(boxes)
    }
}

pub struct Rec {
    session: Session,
    vocab: Vec<String>,
    vocab_len: usize,
}

impl Rec {
    pub fn new(model_path: &str, vocab_path: &str) -> Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        let content = fs::read_to_string(vocab_path)?;
        let mut vocab = vec!["blank".to_string()];
        for line in content.lines() {
            vocab.push(line.trim().to_string());
        }
        vocab.push(" ".to_string());
        let vocab_len = vocab.len();
        Ok(Self {
            session,
            vocab,
            vocab_len,
        })
    }

    pub fn run(&self, img: &DynamicImage) -> Result<String> {
        let model_h = 48;
        let model_w = 320;
        let (w, h) = img.dimensions();
        let ratio = w as f32 / h as f32;
        let new_w = (model_h as f32 * ratio) as u32;
        let new_w = new_w.min(model_w);
        let resized = img.resize_exact(new_w, model_h, FilterType::Triangle);
        let mut input_tensor = Array4::<f32>::zeros((1, 3, model_h as usize, model_w as usize));
        for y in 0..model_h {
            for x in 0..new_w {
                let pixel = resized.get_pixel(x, y);
                let rgb = pixel.to_rgb();
                // BGR conversion for Rec
                let bgr = [rgb[2], rgb[1], rgb[0]];
                for c in 0..3 {
                    let val = bgr[c] as f32 / 255.0;
                    let norm = (val - 0.5) / 0.5;
                    input_tensor[[0, c as usize, y as usize, x as usize]] = norm;
                }
            }
        }

        // Pass (shape, slice) tuple to Value::from_array
        let shape = input_tensor.shape();
        let slice = input_tensor
            .as_slice()
            .ok_or(anyhow::anyhow!("Input tensor is not contiguous"))?;
        let input_value = Value::from_array((shape, slice))?;
        let inputs = inputs!["x" => input_value]?;
        let outputs = self.session.run(inputs)?;

        let tensor = outputs[0].try_extract_tensor::<f32>()?;
        let shape = tensor.shape();
        let seq_len = shape[1];
        let num_classes = shape[2];

        // Convert to view for indexing
        let output_view = tensor.view();

        let mut indices = Vec::new();
        for i in 0..seq_len {
            let mut max_val = -f32::INFINITY;
            let mut max_idx = 0;
            for j in 0..num_classes {
                let val = output_view[[0, i, j]];
                if val > max_val {
                    max_val = val;
                    max_idx = j;
                }
            }
            indices.push(max_idx);
        }
        let mut res = String::new();
        let mut last_idx = usize::MAX;
        for idx in indices {
            if idx != last_idx {
                if idx != 0 && idx < self.vocab_len {
                    res.push_str(&self.vocab[idx]);
                }
                last_idx = idx;
            }
        }
        Ok(res)
    }
}
