use crate::types::{DecodedImage, Point};
use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, DVector, SVD};
use ndarray::{Array1, Array2, Array4};
use ort::value::{Tensor, Value};
use std::arch::x86_64::{_mm_loadu_si128, _mm_storeu_si128};
use std::cmp::min;
use std::collections::VecDeque;

/// 直接解码图像字节数据为DecodedImage结构体    
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage> {
    let img = if bytes.len() >= 4 {
        match &bytes[0..4] {
            b"\xFF\xD8\xFF" => {
                // JPEG格式
                image::load_from_memory_with_format(bytes, image::ImageFormat::Jpeg)?
            }
            b"\x89PNG" => {
                // PNG格式
                image::load_from_memory_with_format(bytes, image::ImageFormat::Png)?
            }
            b"RIFF" if bytes.len() >= 12 && &bytes[8..12] == b"WEBP" => {
                // WebP格式
                image::load_from_memory_with_format(bytes, image::ImageFormat::WebP)?
            }
            b"BM" => {
                // BMP格式
                image::load_from_memory_with_format(bytes, image::ImageFormat::Bmp)?
            }
            _ => image::load_from_memory(bytes)?,
        }
    } else {
        image::load_from_memory(bytes)?
    };
    let (width, height) = (img.width(), img.height());
    let rgb_data = match img {
        image::DynamicImage::ImageRgb8(rgb_img) => rgb_img.into_raw(),
        image::DynamicImage::ImageRgba8(rgba_img) => {
            let rgba_data = rgba_img.into_raw();
            let mut rgb_data = Vec::with_capacity(width as usize * height as usize * 3);
            for chunk in rgba_data.chunks_exact(4) {
                rgb_data.extend_from_slice(&chunk[0..3]);
            }
            rgb_data
        }
        _ => {
            let rgb_img = img.to_rgb8();
            rgb_img.into_raw() // into_raw()比to_vec()更高效
        }
    };
    let format = DecodedImage::detect_format(bytes);
    Ok(DecodedImage::new(width, height, rgb_data, format))
}

pub struct ImageBufferPool {
    buffers: VecDeque<Vec<u8>>,
    max_size: usize,
}
impl ImageBufferPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffers: VecDeque::with_capacity(max_size),
            max_size,
        }
    }
    pub fn get_buffer(&mut self, size: usize) -> Vec<u8> {
        if let Some(mut buffer) = self.buffers.pop_front() {
            if buffer.capacity() >= size {
                buffer.clear();
                buffer.resize(size, 0);
                return buffer;
            }
        }
        Vec::with_capacity(size)
    }
    pub fn return_buffer(&mut self, buffer: Vec<u8>) {
        if self.buffers.len() < self.max_size {
            self.buffers.push_back(buffer);
        }
    }
}

#[inline(always)]
unsafe fn resize_nearest_simd(
    src: &[u8],
    sw: usize,
    sh: usize,
    dst: &mut [u8],
    dw: usize,
    dh: usize,
) {
    // 最近邻映射表预计算
    let mut map_x = vec![0usize; dw];
    let mut map_y = vec![0usize; dh];

    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for i in 0..dw {
        map_x[i] = min((i as f32 * scale_x) as usize, sw - 1);
    }
    for i in 0..dh {
        map_y[i] = min((i as f32 * scale_y) as usize, sh - 1);
    }

    let mut di = 0;
    for y in 0..dh {
        let sy = map_y[y];
        let src_row = sy * sw * 3;

        let mut x = 0;
        while x + 16 < dw {
            let mut temp = [0u8; 48];

            for i in 0..16 {
                let sx = map_x[x + i];
                let si = src_row + sx * 3;
                temp[i * 3] = src[si];
                temp[i * 3 + 1] = src[si + 1];
                temp[i * 3 + 2] = src[si + 2];
            }

            let p1 = _mm_loadu_si128(temp[0..16].as_ptr() as *const _);
            let p2 = _mm_loadu_si128(temp[16..32].as_ptr() as *const _);
            let p3 = _mm_loadu_si128(temp[32..48].as_ptr() as *const _);

            _mm_storeu_si128(dst[di..].as_mut_ptr() as *mut _, p1);
            _mm_storeu_si128(dst[di + 16..].as_mut_ptr() as *mut _, p2);
            _mm_storeu_si128(dst[di + 32..].as_mut_ptr() as *mut _, p3);

            di += 48;
            x += 16;
        }

        while x < dw {
            let sx = map_x[x];
            let si = src_row + sx * 3;
            dst[di] = src[si];
            dst[di + 1] = src[si + 1];
            dst[di + 2] = src[si + 2];
            di += 3;
            x += 1;
        }
    }
}

pub fn letter_box_transform(
    rgb: &[u8],
    width: usize,
    height: usize,
    target_w: usize,
    target_h: usize,
) -> (usize, usize, Vec<u8>, f32, usize, usize) {
    let r = f32::min(
        target_w as f32 / width as f32,
        target_h as f32 / height as f32,
    );

    let new_w = (width as f32 * r).round() as usize;
    let new_h = (height as f32 * r).round() as usize;

    let left = (target_w - new_w) / 2;
    let top = (target_h - new_h) / 2;

    let mut out = vec![114u8; target_w * target_h * 3];

    let mut resized = vec![0u8; new_w * new_h * 3];

    unsafe {
        resize_nearest_simd(rgb, width, height, &mut resized, new_w, new_h);
    }

    for y in 0..new_h {
        let dst_row = (top + y) * target_w * 3 + left * 3;
        let src_row = y * new_w * 3;
        out[dst_row..dst_row + new_w * 3].copy_from_slice(&resized[src_row..src_row + new_w * 3]);
    }

    (target_w, target_h, out, r, left, top)
}

/// 直接处理RGB字节数据的仿射变换，使用最近邻插值
pub fn apply_affine_transform_bytes(
    src_data: &[u8],
    src_w: u32,
    src_h: u32,
    transform: &[[f32; 3]; 2],
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    let [a, b, c] = transform[0];
    let [d, e, f] = transform[1];
    // 预计算常量
    let src_w_minus1 = src_w as f32 - 1.0;
    let src_h_minus1 = src_h as f32 - 1.0;
    // 创建目标图像数据 - 预分配并初始化为0
    let mut dest_data = vec![0u8; (width * height * 3) as usize];
    // 批量处理优化
    for y in 0..height {
        let y_f = y as f32;
        let dest_row_offset = (y * width * 3) as usize;
        for x in 0..width {
            let x_f = x as f32;
            // 计算源坐标
            let src_x = a * x_f + b * y_f + c;
            let src_y = d * x_f + e * y_f + f;
            // 快速边界检查
            if src_x < 0.0 || src_x >= src_w_minus1 || src_y < 0.0 || src_y >= src_h_minus1 {
                continue; // 已经初始化为0，无需再次设置
            }
            // 最近邻插值 - 使用更快的转换
            let src_x_i = src_x as u32;
            let src_y_i = src_y as u32;
            // 计算源索引 - 避免中间转换
            let src_idx = (src_y_i * src_w + src_x_i) as usize * 3;
            let dest_idx = dest_row_offset + (x as usize) * 3;
            // 批量复制3个字节
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src_data.as_ptr().add(src_idx),
                    dest_data.as_mut_ptr().add(dest_idx),
                    3,
                );
            }
        }
    }
    Ok(dest_data)
}

/// 直接返回RGB字节数据的人脸对齐
pub fn face_align(
    src_data: &[u8],
    src_w: u32,
    src_h: u32,
    fivepoint: &[Point],
) -> Result<(u32, u32, Vec<u8>)> {
    let ref_points = [
        [30.29459953 + 8.0, 51.69630051], // 左眼
        [65.53179932 + 8.0, 51.50139999], // 右眼
        [48.02519989 + 8.0, 71.73660278], // 鼻子
        [33.54930115 + 8.0, 92.36550140], // 左嘴角
        [62.72990036 + 8.0, 92.20410156], // 右嘴角
    ];
    // 计算仿射变换矩阵
    let transform_matrix = affine_transform_svd(fivepoint, &ref_points)?;
    // 应用仿射变换
    let aligned_data =
        apply_affine_transform_bytes(src_data, src_w, src_h, &transform_matrix, 112, 112)?;
    Ok((112, 112, aligned_data))
}

/// 直接将RGB字节数据转换为NCHW格式的Tensor
pub fn rgb_bytes_to_tensor(rgb_data: &[u8], width: u32, height: u32) -> Result<Value> {
    let width = width as usize;
    let height = height as usize;
    // 预分配向量
    let mut data = Vec::with_capacity(width * height * 3);
    // 按NCHW顺序填充：R通道 -> G通道 -> B通道
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3 + channel;
                data.push(rgb_data[idx] as f32 / 255.0);
            }
        }
    }
    // 创建ndarray数组 (1, 3, height, width)
    let array = Array4::from_shape_vec((1, 3, height, width), data)?;
    // 转换为ORT的Value
    let tensor = Tensor::from_array(array)?;
    Ok(tensor.into())
}

/// 直接处理112x112的RGB字节数据
pub fn preprocess_bytes(rgb_data: &[u8], width: u32, height: u32) -> Result<Value> {


    // 预分配精确大小的向量
    let size = (3 * 112 * 112) as usize;
    let mut data = Vec::with_capacity(size);

    // 优化：预先计算常量
    let inv_127_5 = 1.0 / 127.5;
    let width_height = 112;

    // 直接按NCHW顺序处理
    for c in 0..3 {
        for y in 0..width_height {
            let y_offset = y * width_height;
            for x in 0..width_height {
                let idx = (y_offset + x) * 3 + c;
                // 归一化到[-1,1]范围 - 使用预计算的常量
                data.push((rgb_data[idx] as f32 - 127.5) * inv_127_5);
            }
        }
    }
    // 创建ndarray数组 (1, 3, 112, 112)
    let array = Array4::from_shape_vec((1, 3, 112, 112), data)?;
    // 转换为ORT的Value
    let tensor = Tensor::from_array(array)?;

    Ok(tensor.into())
}
// 5. 数学计算模块
pub fn box_iou(box1: &Array2<f32>, box2: &Array2<f32>) -> f32 {
    let x1 = box1[(0, 0)];
    let y1 = box1[(0, 1)];
    let x2 = box1[(0, 2)];
    let y2 = box1[(0, 3)];
    let x1_b = box2[(0, 0)];
    let y1_b = box2[(0, 1)];
    let x2_b = box2[(0, 2)];
    let y2_b = box2[(0, 3)];
    // 计算相交区域
    let inter_x1 = x1.max(x1_b);
    let inter_y1 = y1.max(y1_b);
    let inter_x2 = x2.min(x2_b);
    let inter_y2 = y2.min(y2_b);
    if inter_x2 <= inter_x1 || inter_y2 <= inter_y1 {
        return 0.0;
    }
    // 计算面积
    let inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1);
    let area1 = (x2 - x1) * (y2 - y1);
    let area2 = (x2_b - x1_b) * (y2_b - y1_b);
    // 计算IoU
    inter_area / (area1 + area2 - inter_area + 1e-6)
}

/// 非极大值抑制
pub fn nms_numpy(boxes: &Array2<f32>, scores: &Array1<f32>, iou_thres: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut keep = Vec::new();
    while !order.is_empty() {
        let i = order[0];
        keep.push(i);
        if order.len() == 1 {
            break;
        }
        let current_box = boxes.row(i).to_owned().insert_axis(ndarray::Axis(0));
        let mut ious = Vec::new();
        for &j in &order[1..] {
            let other_box = boxes.row(j).to_owned().insert_axis(ndarray::Axis(0));
            ious.push(box_iou(&current_box, &other_box));
        }
        let mut new_order = Vec::new();
        for (idx, &iou) in ious.iter().enumerate() {
            if iou <= iou_thres {
                new_order.push(order[idx + 1]);
            }
        }
        order = new_order;
    }
    keep
}
/// 使用SVD计算仿射变换矩阵
pub fn affine_transform_svd(
    src_points: &[Point],
    dst_points: &[[f32; 2]],
) -> Result<[[f32; 3]; 2]> {
    let mut a_data = Vec::with_capacity(10 * 6);
    let mut b_data = Vec::with_capacity(10);
    for i in 0..5 {
        let x = src_points[i].x as f64;
        let y = src_points[i].y as f64;
        let u = dst_points[i][0] as f64;
        let v = dst_points[i][1] as f64;

        // 第一行：x y 1 0 0 0
        a_data.extend_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0]);
        b_data.push(u);

        // 第二行：0 0 0 x y 1
        a_data.extend_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0]);
        b_data.push(v);
    }
    // 使用nalgebra的SVD求解最小二乘问题
    let a_matrix = DMatrix::from_row_slice(10, 6, &a_data);
    let b_vector = DVector::from_vec(b_data);
    // 计算SVD，需要指定是否计算U和V矩阵
    let svd = SVD::new(a_matrix.clone(), true, true);
    // 计算伪逆 A^+ = V * S^+ * U^T
    let u = svd.u.unwrap();
    let sigma = svd.singular_values;
    let v_t = svd.v_t.unwrap();
    // 计算S的伪逆
    let mut sigma_inv = DMatrix::zeros(6, 6);
    for i in 0..6.min(sigma.len()) {
        if sigma[i] > 1e-10 {
            sigma_inv[(i, i)] = 1.0 / sigma[i];
        }
    }
    // 计算伪逆矩阵
    let v = v_t.transpose(); // 获取V矩阵
    let u_t = u.transpose(); // 获取U^T矩阵
    let a_pinv = v * sigma_inv * u_t;
    // 求解 x = A^+ * b
    let x = a_pinv * b_vector;
    // 提取仿射变换参数
    let a = x[0] as f32;
    let b = x[1] as f32;
    let c = x[2] as f32;
    let d = x[3] as f32;
    let e = x[4] as f32;
    let f = x[5] as f32;
    // 计算反向变换
    let det = a * e - b * d;
    let inv_a = e / det;
    let inv_b = -b / det;
    let inv_c = (b * f - c * e) / det;
    let inv_d = -d / det;
    let inv_e = a / det;
    let inv_f = (c * d - a * f) / det;
    Ok([
        [inv_a, inv_b, inv_c], // 反向变换参数
        [inv_d, inv_e, inv_f], // 反向变换参数
    ])
}
