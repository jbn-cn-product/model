use crate::types::{
    get_session_pool, BoundingBox, DecodedImage, FaceDetection, FaceFeature, Point, ModelType,
};
use crate::utils::{
    face_align, letter_box_transform, nms_numpy, preprocess_bytes, rgb_bytes_to_tensor,
};
use anyhow::Result;
use ndarray::{Array1, Array2, Array3, ArrayViewD};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

/// 提取人脸特征向量
pub async fn feature_extract(
    decoded_image: &DecodedImage,
    fivepoint: &[Point],
) -> Result<FaceFeature> {
    let session_pool = get_session_pool(ModelType::FaceRec)?;
    let rgb_data = &decoded_image.rgb_data;
    let width = decoded_image.width;
    let height = decoded_image.height;

    let (aligned_w, aligned_h, aligned_data) = face_align(rgb_data, width, height, fivepoint)?;
    let input_tensor = preprocess_bytes(&aligned_data, aligned_w, aligned_h)?;

    let embeddings_vec = session_pool
        .execute(|session| {
            let outputs = session.run(inputs![input_tensor])?;
            let embeddings: ArrayViewD<f32> = outputs[0].try_extract_array()?;
            Ok(embeddings.as_slice().unwrap().to_vec())
        })
        .await?;

    let norm_sq: f32 = embeddings_vec.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    let normalized_features: Vec<f32> = embeddings_vec.iter().map(|x| x / norm).collect();

    Ok(FaceFeature::new(normalized_features))
}

/// 人脸检测器
///
/// 基于YOLOv5模型的人脸检测器，能够同时检测人脸位置和5个关键点
///
/// # 字段
///
/// * `session` - ONNX推理会话
/// * `img_size` - 输入图像尺寸
/// * `buffer_pool` - 图像缓冲池
pub struct FaceDetector {
    session: Session,
    img_size: (u32, u32),
}
impl FaceDetector {
    /// 创建新的人脸检测器=
    ///
    /// 调用 `init_model_config()` 初始化模型路径
    ///
    /// # 返回值
    ///
    /// 返回新的FaceDetector实例
    ///
    /// # 错误
    ///
    /// 如果模型未初始化或加载失败会返回错误
    //=
    pub fn new() -> Result<FaceDetector> {
        use crate::types::{MODEL_CONFIG, ModelType};

        let path = MODEL_CONFIG
            .get()
            .ok_or_else(|| anyhow::anyhow!("Model config not initialized. Call init_model_config() first."))?
            .get_model_path(ModelType::FaceDet);

        let session = Session::builder()?
            .with_execution_providers([
                ort::execution_providers::CPUExecutionProvider::default().build()
            ])?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&path)?;
        Ok(FaceDetector {
            session,
            img_size: (640, 640),
        })
    }

    /// 非极大值抑制和后处理
    ///
    /// 对模型输出进行后处理，包括：
    /// - 置信度筛选
    /// - 非极大值抑制
    /// - 边界框和关键点解析
    ///
    /// # 参数
    ///
    /// * `prediction` - 模型原始输出
    /// * `conf_thres` - 置信度阈值
    /// * `iou_thres` - IoU阈值
    ///
    /// # 返回值
    ///
    /// 返回处理后的人脸检测结果列表
    fn no_max(
        &self,
        prediction: &Array3<f32>,
        conf_thres: f32,
        iou_thres: f32,
    ) -> Vec<FaceDetection> {
        let mut output = Vec::new();
        for img_idx in 0..prediction.dim().0 {
            let img_pred =
                prediction.slice_axis(ndarray::Axis(0), ndarray::Slice::from(img_idx..=img_idx));
            let mut valid_indices = Vec::new();
            for i in 0..img_pred.dim().1 {
                if img_pred[(0, i, 4)] > conf_thres {
                    valid_indices.push(i);
                }
            }
            if valid_indices.is_empty() {
                continue;
            }
            let mut x = Array3::from_shape_fn(
                (1, valid_indices.len(), img_pred.dim().2),
                |(_, new_idx, j)| img_pred[(0, valid_indices[new_idx], j)],
            );
            for i in 0..x.dim().1 {
                let conf = x[(0, i, 4)];
                for j in 15..x.dim().2 {
                    x[(0, i, j)] *= conf;
                }
            }
            let boxes = Array2::from_shape_fn((x.dim().1, 4), |(i, j)| {
                let center_x = x[(0, i, 0)];
                let center_y = x[(0, i, 1)];
                let width = x[(0, i, 2)];
                let height = x[(0, i, 3)];
                match j {
                    0 => center_x - width / 2.0,  // x1
                    1 => center_y - height / 2.0, // y1
                    2 => center_x + width / 2.0,  // x2
                    3 => center_y + height / 2.0, // y2
                    _ => unreachable!(),
                }
            });
            let (conf_scores, class_indices): (Vec<_>, Vec<_>) = (0..x.dim().1)
                .map(|i| {
                    let conf = x[(0, i, 4)];
                    let plate_conf = x[(0, i, 15)];
                    let face_conf = x[(0, i, 16)];
                    let class_idx = if face_conf > plate_conf { 1 } else { 0 };
                    (conf, class_idx)
                })
                .unzip();
            // 再次筛选置信度
            let mut filtered_data = Vec::new();
            for i in 0..x.dim().1 {
                if conf_scores[i] > conf_thres {
                    let mut row = Vec::with_capacity(16);
                    // 边界框坐标
                    for j in 0..4 {
                        row.push(boxes[(i, j)]);
                    }
                    // 置信度
                    row.push(conf_scores[i]);
                    // 关键点坐标（5-15列、
                    for j in 5..15 {
                        row.push(x[(0, i, j)]);
                    }
                    // 类别索引
                    row.push(class_indices[i] as f32);
                    filtered_data.push(row);
                }
            }
            let final_boxes = Array2::from_shape_vec(
                (filtered_data.len(), 4),
                filtered_data
                    .iter()
                    .flat_map(|row| row.iter().take(4).cloned())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let final_scores =
                Array1::from_vec(filtered_data.iter().map(|row| row[4]).collect::<Vec<_>>());
            let keep = nms_numpy(&final_boxes, &final_scores, iou_thres);
            for &i in &keep {
                let xyxy = final_boxes.row(i);
                let conf = final_scores[i];
                let landmarks = (0..5)
                    .map(|k| {
                        Point::new(
                            filtered_data[i][5 + k * 2] as i32,
                            filtered_data[i][5 + k * 2 + 1] as i32,
                        )
                    })
                    .collect();
                // 获取原始类别索引
                let class_idx = filtered_data[i][15] as i32;
                output.push(FaceDetection::new(
                    BoundingBox::new(
                        xyxy[0] as i32,
                        xyxy[1] as i32,
                        xyxy[2] as i32,
                        xyxy[3] as i32,
                    ),
                    landmarks,
                    conf,
                    class_idx,
                ));
            }
        }
        output
    }

    /// 检测图像中的人脸
    ///
    /// 对输入图像进行人脸检测，返回检测到的人脸列表，每个人脸包含边界框、关键点和置信度信息。
    ///
    /// # 处理流程
    ///
    /// 1. 图像预处理（letterbox变换）
    /// 2. ONNX模型推理
    /// 3. 后处理（NMS、坐标映射）
    /// 4. 返回检测结果
    ///
    /// # 参数
    ///
    /// * `decoded_image` - 已解码的RGB图像数据
    ///
    /// # 返回值
    ///
    /// 返回检测到的人脸列表，如果未检测到人脸则返回空列表。

    pub fn detect(&mut self, decoded_image: &DecodedImage) -> Vec<FaceDetection> {
        // 直接使用已解码的RGB数据
        let rgb_data = &decoded_image.rgb_data;
        let width = decoded_image.width;
        let height = decoded_image.height;

        let (target_w, target_h) = self.img_size;

        let (padded_width, padded_height, padded_data, r, left, top) = letter_box_transform(
            rgb_data,
            width as usize,
            height as usize,
            target_w as usize,
            target_h as usize,
        );
        // 转换为Tensor
        let Ok(input_tensor) =
            rgb_bytes_to_tensor(&padded_data, padded_width as u32, padded_height as u32)
        else {
            println!("face detect error 0");
            return vec![];
        };

        let Ok(outputs) = self.session.run(inputs![input_tensor]) else {
            println!("face detect error 1");
            return vec![];
        };
        let Ok(pred) = outputs[0].try_extract_array() else {
            println!("face detect error 2");
            return vec![];
        };
        let pred_shape = pred.shape();
        let dim0 = pred_shape[0] as usize;
        let dim1 = pred_shape[1] as usize;
        let dim2 = if pred_shape.len() == 4 {
            (pred_shape[2] * pred_shape[3]) as usize
        } else {
            pred_shape[2] as usize
        };
        let Ok(pred_3d) =
            Array3::from_shape_vec((dim0, dim1, dim2), pred.as_slice().unwrap().to_vec())
        else {
            println!("face detect error 3");
            return vec![];
        };

        drop(outputs);
        let results = self.no_max(&pred_3d, 0.3, 0.5);
        let mut final_results = Vec::new();
        for detection in results {
            let bbox = detection.bbox;
            let landmarks = detection.fivepoint;
            let scaled_bbox = BoundingBox::new(
                ((bbox.x1 as f32 - left as f32) / r) as i32,
                ((bbox.y1 as f32 - top as f32) / r) as i32,
                ((bbox.x2 as f32 - left as f32) / r) as i32,
                ((bbox.y2 as f32 - top as f32) / r) as i32,
            );
            let mut scaled_landmarks = Vec::new();
            for point in landmarks {
                scaled_landmarks.push(Point::new(
                    ((point.x as f32 - left as f32) / r) as i32,
                    ((point.y as f32 - top as f32) / r) as i32,
                ));
            }
            final_results.push(FaceDetection::new(
                scaled_bbox,
                scaled_landmarks,
                detection.conf,
                detection.class,
            ));
        }
        final_results
    }
}
