use face_rs::{decode_image, feature_extract, FaceSearchIndex, FaceDetector};
use image::{DynamicImage, ImageBuffer};

#[tokio::main]
async fn main() {
    // 初始化模型配置
    if let Err(e) = face_rs::init_model_config(
        "./models/car_face_det.onnx".to_string(),
        "./models/face_rec.onnx".to_string(),
    ) {
        println!("模型配置失败: {}", e);
        return;
    }

    // 创建人脸检测器 
    let mut detector = FaceDetector::new()
        .expect("Failed to create face detector");  

    let image_paths = vec![
        "test_img/9宫脸.png",
    ];

    for (_idx, image_path) in image_paths.iter().enumerate() {
        // 读取图像
        let image_bytes = match std::fs::read(image_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                println!("读取图像 {} 失败: {}", image_path, e);
                return;
            }
        };

        // 解码图像
        let decoded_image = match decode_image(&image_bytes) {
            Ok(decoded) => decoded,
            Err(e) => {
                println!("解码失败: {}", e);
                return;
            }
        };

       
        let detections = detector.detect(&decoded_image);
        println!("检测到 {} 个目标", detections.len());

        // 过滤人脸 (class == 1)
        let face_detections: Vec<_> = detections.iter()
            .filter(|d| d.class == 1)
            .collect();

        println!("其中人脸 {} 张", face_detections.len());

        if face_detections.is_empty() {
            println!("没有检测到人脸，跳过");
            continue;
        }

        let feature_tasks: Vec<_> = face_detections.iter().enumerate()
            .map(|(i, detection)| {
                let decoded_clone = decoded_image.clone();
                let fivepoint = detection.fivepoint.clone();

                async move {
                    println!("处理第 {} 张人脸，置信度: {:.4}", i + 1, detection.conf);
                    feature_extract(&decoded_clone, &fivepoint).await
                }
            })
            .collect();

        let features: Vec<Result<_, _>> = futures::future::join_all(feature_tasks).await;

    
        for (i, detection) in face_detections.iter().enumerate() {
            let (aligned_w, aligned_h, aligned_data) = match face_rs::utils::face_align(
                &decoded_image.rgb_data,
                decoded_image.width,
                decoded_image.height,
                &detection.fivepoint,
            ) {
                Ok(result) => result,
                Err(e) => {
                    println!("人脸对齐 {} 失败: {}", i + 1, e);
                    continue;
                }
            };

            let rgb_img = ImageBuffer::from_raw(aligned_w, aligned_h, aligned_data)
                .expect("Failed to create image buffer");
            let dynamic_img = DynamicImage::ImageRgb8(rgb_img);
            let output_path = format!("./aligned/aligned_face_{}.jpg", i + 1);
            if let Err(e) = dynamic_img.save(&output_path) {
                println!("保存对齐图像失败: {}", e);
            }
        }

     
        let valid_features: Vec<_> = features.iter()
            .filter_map(|f| f.as_ref().ok())
            .cloned()
            .collect();


        let mut search_index = FaceSearchIndex::new()
            .expect("创建人脸搜索索引失败");
        search_index.add_batch(&valid_features)
            .expect("Failed to add face to search index");

        if let Some(query_feature) = valid_features.get(1) {
            let search_results = search_index.search(query_feature, face_detections.len())
                .expect("Failed to search similar faces");
            for (i, (id, similarity)) in search_results.iter().enumerate() {
                println!("FaceSearchIndex 结果{}: ID={}, 相似度={:.6}", i + 1, id, similarity);
            }
        }

  
        if features.len() > 1 {
          
            for (i, feature_result_i) in features.iter().enumerate() {
                if let Ok(feature_i) = feature_result_i {
                    for (j, feature_result_j) in features.iter().enumerate() {
                        if j > i {
                            if let Ok(feature_j) = feature_result_j {
                                let similarity = feature_i.similarity(feature_j);
                                println!("人脸 {} 与人脸 {} 相似度: {:.6}", i + 1, j + 1, similarity);
                            }
                        }
                    }
                }
            }
        }
    }
}
