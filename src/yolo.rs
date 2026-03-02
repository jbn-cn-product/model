use crate::{Context, LetterBoxInfo, Object, TensorData, rknn_ffi};
use image::Rgb;
use std::ffi::c_void;
use std::fs;
use std::mem;
use std::ptr;

const PREFIX: &str = "yolo";

pub struct Yolo {
    context: rknn_ffi::rknn_context,
    input_attrs: Vec<rknn_ffi::rknn_tensor_attr>,
    output_attrs: Vec<rknn_ffi::rknn_tensor_attr>,
    io_num: rknn_ffi::rknn_input_output_num,
    model_width: i32,
    model_height: i32,
    conf_threshold: f32,
}

impl Context for Yolo {
    fn pre(&self, image_bytes: &[u8]) -> Result<(TensorData, LetterBoxInfo), String> {
        // Decode image
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| format!("{} image decode err: {}", PREFIX, e))?;

        let width = img.width();
        let height = img.height();
        let rgb_img = img.to_rgb8();

        let org_w = width as i32;
        let org_h = height as i32;
        let img_w = width as f32;
        let img_h = height as f32;
        let model_w = self.model_width as f32;
        let model_h = self.model_height as f32;

        let scale = (model_w / img_w).min(model_h / img_h);
        let new_w = (img_w * scale) as u32;
        let new_h = (img_h * scale) as u32;

        let pad_x = (self.model_width as u32 - new_w) / 2;
        let pad_y = (self.model_height as u32 - new_h) / 2;

        // Resize
        let resized_img =
            image::imageops::resize(&rgb_img, new_w, new_h, image::imageops::FilterType::Nearest);

        // Pad
        let mut padded_img = image::ImageBuffer::from_pixel(
            self.model_width as u32,
            self.model_height as u32,
            Rgb([114, 114, 114]),
        );

        image::imageops::overlay(&mut padded_img, &resized_img, pad_x as i64, pad_y as i64);

        // To UInt8 [0-255] (Raw pixel values)
        // Optimization: Use raw bytes directly instead of converting to float
        let input = padded_img.into_raw();

        let info = LetterBoxInfo {
            scale,
            pad_x: pad_x as f32,
            pad_y: pad_y as f32,
            unpad_w: new_w as i32,
            unpad_h: new_h as i32,
            org_w,
            org_h,
        };
        Ok((TensorData::UInt8(input), info))
    }

    fn inference(&self, input_data: &TensorData) -> Result<Vec<Vec<f32>>, String> {
        let (input_ptr, input_size, input_type) = match input_data {
            TensorData::Float(data) => (
                data.as_ptr() as *mut c_void,
                (data.len() * 4) as u32,
                rknn_ffi::_rknn_tensor_type_RKNN_TENSOR_FLOAT32,
            ),
            TensorData::UInt8(data) => (
                data.as_ptr() as *mut c_void,
                data.len() as u32,
                rknn_ffi::_rknn_tensor_type_RKNN_TENSOR_UINT8,
            ),
        };

        if self.io_num.n_input != 1 {
            return Err(format!(
                "{} Expected 1 input, got {}",
                PREFIX, self.io_num.n_input
            ));
        }

        // Set Inputs

        let mut rknn_inputs: Vec<rknn_ffi::rknn_input> = self
            .input_attrs
            .iter()
            .map(|attr| rknn_ffi::rknn_input {
                index: attr.index,
                buf: input_ptr,
                size: input_size,
                pass_through: 0,
                type_: input_type,
                fmt: rknn_ffi::_rknn_tensor_format_RKNN_TENSOR_NHWC,
            })
            .collect();

        unsafe {
            let ret = rknn_ffi::rknn_inputs_set(
                self.context,
                self.io_num.n_input,
                rknn_inputs.as_mut_ptr(),
            );
            if ret != rknn_ffi::RKNN_SUCC as i32 {
                return Err(format!("rknn_inputs_set err={}", ret));
            }
        }

        // Run

        unsafe {
            let ret = rknn_ffi::rknn_run(self.context, ptr::null_mut());
            if ret != rknn_ffi::RKNN_SUCC as i32 {
                return Err(format!("rknn_run failed: {}", ret));
            }
        }

        // Get Outputs

        let mut rknn_outputs: Vec<rknn_ffi::rknn_output> = self
            .output_attrs
            .iter()
            .map(|attr| rknn_ffi::rknn_output {
                want_float: 1,
                is_prealloc: 0,
                index: attr.index,
                buf: ptr::null_mut(),
                size: 0,
            })
            .collect();

        unsafe {
            let ret = rknn_ffi::rknn_outputs_get(
                self.context,
                self.io_num.n_output,
                rknn_outputs.as_mut_ptr(),
                ptr::null_mut(),
            );
            if ret != rknn_ffi::RKNN_SUCC as i32 {
                return Err(format!("rknn_outputs_get failed: {}", ret));
            }
        }

        let outputs: Vec<Vec<f32>> = rknn_outputs
            .iter()
            .map(|output| unsafe {
                std::slice::from_raw_parts(output.buf as *const f32, (output.size / 4) as usize)
                    .to_vec()
            })
            .collect();

        unsafe {
            rknn_ffi::rknn_outputs_release(
                self.context,
                self.io_num.n_output,
                rknn_outputs.as_mut_ptr(),
            );
        }

        Ok(outputs)
    }
}

// Post-processing for End-to-End
impl Yolo {
    /// Main entry point for post-processing
    /// Dispatches to E2E or Raw implementation based on output structure
    pub fn process(
        &self,
        raw_outputs: &[Vec<f32>],
        letterbox_info: &LetterBoxInfo,
    ) -> Result<Vec<Object>, String> {
        let n_out = raw_outputs.len();
        if n_out == 1 {
            self.post_e2e(raw_outputs, letterbox_info)
        } else if n_out == 6 {
            self.post_raw(raw_outputs, letterbox_info)
        } else {
            Err(format!("{} Unsupported output count: {}", PREFIX, n_out))
        }
    }

    /// Post-processing for End-to-End model (1 output: [1, N, 6])
    fn post_e2e(
        &self,
        raw_outputs: &[Vec<f32>],
        letterbox_info: &LetterBoxInfo,
    ) -> Result<Vec<Object>, String> {
        let data = &raw_outputs[0];
        if data.len() % 6 != 0 {
            return Err(format!("{} E2E data len not div by 6", PREFIX));
        }

        let num_dets = data.len() / 6;
        let mut objects = Vec::with_capacity(num_dets);

        for i in 0..num_dets {
            let off = i * 6;
            let score = data[off + 4];
            if score > self.conf_threshold {
                let class_id = data[off + 5] as i32;
                let bbox = self._convert_box(
                    &[data[off], data[off + 1], data[off + 2], data[off + 3]],
                    letterbox_info,
                );

                objects.push(Object {
                    bbox,
                    class_id,
                    confidence: score,
                });
            }
        }

        // E2E usually doesn't need NMS, or minimal NMS. Assuming sorted.
        objects.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(objects)
    }

    /// Post-processing for Raw/Split-Head model (6 outputs)
    /// Structure:
    /// 0: Box(80x80) [1, 4, 80, 80]
    /// 1: Cls(80x80) [1, Cls, 80, 80]
    /// 2: Box(40x40) ...
    fn post_raw(
        &self,
        raw_outputs: &[Vec<f32>],
        letterbox_info: &LetterBoxInfo,
    ) -> Result<Vec<Object>, String> {
        use crate::utils; // Use our optimized utils

        let mut candidates = Vec::new();

        // Strides for the 3 scales
        let strides = [8, 16, 32];
        let grids = [80, 40, 20]; // Assuming 640 input. Logic should be dynamic but simplified here.

        // Iterate over scales
        for i in 0..3 {
            let box_idx = i * 2;
            let cls_idx = i * 2 + 1;

            let box_data = &raw_outputs[box_idx];
            let cls_data = &raw_outputs[cls_idx];

            let grid_h = grids[i];
            let grid_w = grids[i];
            let _stride = strides[i];

            let cls_len = cls_data.len();
            let area = grid_h * grid_w;
            let num_classes = cls_len / area;

            // Validate sizes
            // box_data len should be 4 * area
            if box_data.len() != 4 * area {
                return Err(format!("{} Box tensor {} size mismatch", PREFIX, box_idx));
            }

            // Iterate spatial grid
            for h in 0..grid_h {
                for w in 0..grid_w {
                    let spatial_idx = h * grid_w + w;

                    // Find Post-Sigmoid Class Score
                    // Cls data is usually [1, C, H, W] -> Flattened: [C, H, W] or [H, W, C]?
                    // Dump says NCHW. So: Channel 0 is full grid, Channel 1 is full grid...
                    // Score(c, h, w) = data[c * area + spatial_idx]

                    let mut max_cls_score = 0.0f32;
                    let mut max_cls_id = -1;

                    for c in 0..num_classes {
                        // Simple linear scan for max class (can be optimized with SIMD)
                        let s = cls_data[c * area + spatial_idx];
                        // If model doesn't include sigmoid, apply it here:
                        // let s = utils::fast_sigmoid(s);
                        // Check metadata: usually NPU models bake sigmoid in.

                        if s > max_cls_score {
                            max_cls_score = s;
                            max_cls_id = c as i32;
                        }
                    }

                    if max_cls_score > self.conf_threshold {
                        // Decode Box
                        // Box data NCHW: 4 channels.
                        // [0*area + idx] = d0 (x or l)
                        // [1*area + idx] = d1 (y or t)
                        // ...

                        // RKNN output1 shape [1, 4, 80, 80].
                        // Values are usually xywh or lrtb relative to anchor??
                        // If output is [1, 4, ..], it's likely [dx, dy, dw, dh] * stride?
                        // Or if it's DFL-pre-decoded, it might be coordinates?

                        // Assuming "reg_max=1" export (no DFL, just regression)
                        // It usually outputs (cx, cy, w, h) relative to image or grid.

                        let d0 = box_data[0 * area + spatial_idx];
                        let d1 = box_data[1 * area + spatial_idx];
                        let d2 = box_data[2 * area + spatial_idx];
                        let d3 = box_data[3 * area + spatial_idx];

                        // Standard YOLOv8 export without DFL often gives:
                        // cx = (d0 + w) * stride ??
                        // No, if shape is 4, it's [cx, cy, w, h] usually.
                        // Let's assume standard center-xywh format for now.

                        // We need to verify if it's normalized or pixel coords.
                        // Assuming pixel coords based on typical NPU exports.

                        // Convert cx,cy,w,h -> x1,y1,x2,y2
                        let cx = d0;
                        let cy = d1;
                        let w = d2;
                        let h = d3;

                        let x1 = cx - w * 0.5;
                        let y1 = cy - h * 0.5;
                        let x2 = cx + w * 0.5;
                        let y2 = cy + h * 0.5;

                        let bbox = self._convert_box(&[x1, y1, x2, y2], letterbox_info);

                        candidates.push(Object {
                            bbox,
                            class_id: max_cls_id,
                            confidence: max_cls_score,
                        });
                    }
                }
            }
        }

        // NMS
        let keep = utils::non_max_suppression(&mut candidates, 0.45); // 0.45 IoU threshold
        let final_objs = keep.into_iter().map(|i| candidates[i].clone()).collect();

        Ok(final_objs)
    }
}

impl Yolo {
    pub fn new(core_mask: u32, model_path: &str, conf_threshold: f32) -> Result<Self, String> {
        let res_model = fs::read(model_path);
        if let Err(e) = res_model {
            return Err(format!("model read err={}", e));
        }
        let model = res_model.unwrap();

        let mut context: rknn_ffi::rknn_context = 0;

        // init
        unsafe {
            let ret = rknn_ffi::rknn_init(
                &mut context,
                model.as_ptr() as *mut c_void,
                model.len() as u32,
                0,
                ptr::null_mut(),
            );
            if ret != rknn_ffi::RKNN_SUCC as i32 {
                return Err(format!("rknn_init err={}", ret));
            }
        }

        // core mask
        unsafe {
            let ret = rknn_ffi::rknn_set_core_mask(context, core_mask);
            if ret != rknn_ffi::RKNN_SUCC as i32 {
                rknn_ffi::rknn_destroy(context);
                return Err(format!("rknn_set_core_mask err={}", ret));
            }
        }

        // query in/out num
        let mut io_num: rknn_ffi::rknn_input_output_num = unsafe { mem::zeroed() };
        unsafe {
            let ret = rknn_ffi::rknn_query(
                context,
                rknn_ffi::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM,
                &mut io_num as *mut _ as *mut c_void,
                mem::size_of::<rknn_ffi::rknn_input_output_num>() as u32,
            );
            if ret != rknn_ffi::RKNN_SUCC as i32 {
                rknn_ffi::rknn_destroy(context);
                return Err(format!("rknn_query IN_OUT_NUM err={}", ret));
            }
        }

        // query inputs
        let mut input_attrs = Vec::with_capacity(io_num.n_input as usize);
        for i in 0..io_num.n_input {
            let mut attr: rknn_ffi::rknn_tensor_attr = unsafe { mem::zeroed() };
            attr.index = i;
            unsafe {
                let ret = rknn_ffi::rknn_query(
                    context,
                    rknn_ffi::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    mem::size_of::<rknn_ffi::rknn_tensor_attr>() as u32,
                );
                if ret != rknn_ffi::RKNN_SUCC as i32 {
                    rknn_ffi::rknn_destroy(context);
                    return Err(format!("rknn_query INPUT_ATTR err={}", ret));
                }
            }
            input_attrs.push(attr);
        }

        // query outputs
        let mut output_attrs = Vec::with_capacity(io_num.n_output as usize);
        for i in 0..io_num.n_output {
            let mut attr: rknn_ffi::rknn_tensor_attr = unsafe { mem::zeroed() };
            attr.index = i;
            unsafe {
                let ret = rknn_ffi::rknn_query(
                    context,
                    rknn_ffi::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR,
                    &mut attr as *mut _ as *mut c_void,
                    mem::size_of::<rknn_ffi::rknn_tensor_attr>() as u32,
                );
                if ret != rknn_ffi::RKNN_SUCC as i32 {
                    rknn_ffi::rknn_destroy(context);
                    return Err(format!("rknn_query OUTPUT_ATTR err={}", ret));
                }
            }
            output_attrs.push(attr);
        }

        // Determine Model Dimensions
        let input_attr = &input_attrs[0];
        let (model_width, model_height) =
            if input_attr.fmt == rknn_ffi::_rknn_tensor_format_RKNN_TENSOR_NCHW {
                (input_attr.dims[3] as i32, input_attr.dims[2] as i32)
            } else {
                (input_attr.dims[2] as i32, input_attr.dims[1] as i32)
            };

        println!(
            "{} Init: w={} h={} inputs={} outputs={}",
            PREFIX, model_width, model_height, io_num.n_input, io_num.n_output
        );

        Ok(Self {
            context,
            input_attrs,
            output_attrs,
            io_num,
            model_width,
            model_height,
            conf_threshold,
        })
    }

    fn _convert_box(&self, coords: &[f32; 4], info: &LetterBoxInfo) -> [f32; 4] {
        let x1_u = coords[0] - info.pad_x;
        let y1_u = coords[1] - info.pad_y;
        let x2_u = coords[2] - info.pad_x;
        let y2_u = coords[3] - info.pad_y;
        [
            (x1_u / info.scale).max(0.0).min(info.org_w as f32),
            (y1_u / info.scale).max(0.0).min(info.org_h as f32),
            (x2_u / info.scale).max(0.0).min(info.org_w as f32),
            (y2_u / info.scale).max(0.0).min(info.org_h as f32),
        ]
    }
}

impl Drop for Yolo {
    fn drop(&mut self) {
        unsafe {
            rknn_ffi::rknn_destroy(self.context);
        }
    }
}
