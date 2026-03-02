//! types 模块
//! 包括点坐标、边界框、人脸检测结果、特征向量等基础类型。

use std::sync::{Arc, OnceLock};
use std::collections::VecDeque;
use ort::editor::Model;
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use anyhow::Result;
use ort::session::{Session};
use ort::session::builder::GraphOptimizationLevel;

/// 表示二维平面上的点坐标
/// 
/// # 字段
/// 
/// * `x` - x坐标
/// * `y` - y坐标
/// 
/// # 示例
/// 
/// ```
/// use face_rust::types::Point;
/// let point = Point::new(100, 200);
/// assert_eq!(point.x, 100);
/// assert_eq!(point.y, 200);
/// ```
#[derive(Debug,Clone,Copy)]
pub struct Point{
    pub x:i32,
    pub y:i32,
}
impl Point{
    /// 创建一个新的点
    /// 
    /// # 参数
    /// 
    /// * `x` - x坐标
    /// * `y` - y坐标
    /// 
    /// # 返回值
    /// 
    /// 返回一个新的Point实例
    /// 
    /// # 示例
    /// 
    /// ```
    /// use face_rust::types::Point;
    /// let point = Point::new(10, 20);
    /// assert_eq!(point.x, 10);
    /// assert_eq!(point.y, 20);
    /// ```
    pub fn new(x:i32,y:i32) -> Self{
        Self{x,y}
    }
}

/// 表示矩形边界框
/// 
/// 用于表示人脸检测、物体检测等任务中的边界框区域。
/// 
/// # 字段
/// 
/// * `x1` - 左上角x坐标
/// * `y1` - 左上角y坐标  
/// * `x2` - 右下角x坐标
/// * `y2` - 右下角y坐标
/// 
/// # 示例
/// 
/// ```
/// use face_rust::types::BoundingBox;
/// let bbox = BoundingBox::new(10, 20, 100, 200);
/// assert_eq!(bbox.width(), 90.0);
/// assert_eq!(bbox.heigth(), 180.0);
/// ```
#[derive(Debug,Clone,Copy)]
pub struct BoundingBox{
    pub x1:i32,
    pub y1:i32,
    pub x2:i32,
    pub y2:i32,
}

/// 人脸姿态角度
/// 
/// 表示人脸在三维空间中的姿态，包含俯仰角、偏航角和翻滚角。
/// 
/// # 字段
/// 
/// * `pitch` - 俯仰角（点头角度）
/// * `yaw` - 偏航角（摇头角度）  
/// * `roll` - 翻滚角（侧头角度）
#[derive(Debug, Clone, Copy)]
pub struct FacePoseAngles {
    pub pitch: f32,
    pub yaw: f32,
    pub roll: f32,
}

impl FacePoseAngles {
    /// 创建新的人脸姿态角度
    /// 
    /// # 参数
    /// 
    /// * `pitch` - 俯仰角
    /// * `yaw` - 偏航角
    /// * `roll` - 翻滚角
    /// 
    /// # 返回值
    /// 
    /// 返回新的FacePoseAngles实例
    pub fn new(pitch: f32, yaw: f32, roll: f32) -> Self {
        Self { pitch, yaw, roll }
    }
}

/// ONNX推理会话池
/// 
/// 用于管理多个ONNX推理会话。
/// 
/// # 字段
/// 
/// * `sessions` - 会话队列
/// * `semaphore` - 控制并发访问的信号量
/// * `pool_size` - 池大小
pub struct SessionPool {
    sessions: Arc<AsyncMutex<VecDeque<Session>>>,
    semaphore: Arc<Semaphore>,
    pool_size: usize,
}

impl SessionPool {
    /// 创建新的会话池
    /// 
    /// # 参数
    /// 
    /// * `model_path` - ONNX模型文件路径
    /// * `pool_size` - 池大小（会话数量）
    /// 
    /// # 返回值
    /// 
    /// 返回新的SessionPool实例
    /// 
    /// # 错误
    /// 
    /// 如果模型加载失败会返回错误
    pub fn new(model_path: &str, pool_size: usize) -> Result<Self> {
        let mut sessions = VecDeque::with_capacity(pool_size);
        
        for _ in 0..pool_size {
            let session = Session::builder()?
                .with_execution_providers([
                    ort::execution_providers::CPUExecutionProvider::default().build()
                ])?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(model_path)?;
            sessions.push_back(session);
        }
        
        Ok(Self {
            sessions: Arc::new(AsyncMutex::new(sessions)),
            semaphore: Arc::new(Semaphore::new(pool_size)),
            pool_size,
        })
    }
    
    /// 获取池大小
    /// 
    /// # 返回值
    /// 
    /// 返回池中的会话数量
    pub fn pool_size(&self) -> usize {
        self.pool_size
    }
    
    /// 执行推理操作
    /// 
    /// 从池中获取一个会话，执行指定的操作，然后将会话返回池中。
    /// 
    /// # 参数
    /// 
    /// * `operation` - 要执行的推理操作
    /// 
    /// # 返回值
    /// 
    /// 返回推理操作的结果
    /// 
    /// # 泛型参数
    /// 
    /// * `F` - 操作函数类型
    /// * `R` - 返回值类型
    pub async fn execute<F, R>(&self, operation: F) -> Result<R>
    where
        F: FnOnce(&mut Session) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let _permit = self.semaphore.acquire().await?;
        let mut session = {
            let mut sessions = self.sessions.lock().await;
            sessions.pop_front().ok_or_else(|| anyhow::anyhow!("No session available"))?
        };
        
        let result = operation(&mut session);
        
        {
            let mut sessions = self.sessions.lock().await;
            sessions.push_back(session);
        }
        
        result
    }
}


/// 模型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    FaceDet,
    FaceRec,
}
/// 全局模型加载
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub facedet_path:String,
    pub facerec_path:String,
}
impl ModelConfig{
    /// 创建一个新的 `ModelConfig` 实例。
    ///
    /// # 参数
    /// - `facedet_path`: 人脸检测模型的文件路径。
    /// - `facerec_path`: 人脸识别模型的文件路径。
    ///
    /// # 示例
    /// ```
    /// let config = ModelConfig::new("det.onnx".to_string(), "rec.onnx".to_string());
    /// ```
    pub fn new(facedet_path:String,facerec_path:String)->Self{
        Self{facedet_path,facerec_path}
    }
    pub fn get_model_path(&self,model_type:ModelType)->&str{
        match model_type{
            ModelType::FaceDet => &self.facedet_path,
            ModelType::FaceRec => &self.facerec_path,
        }
    }
}


pub static MODEL_CONFIG: OnceLock<ModelConfig> = OnceLock::new();
static FACE_DET_POOL: OnceLock<SessionPool> = OnceLock::new();
static FACE_REC_POOL: OnceLock<SessionPool> = OnceLock::new();

pub fn init_model_config(facedet_path:String,facerec_path:String) -> Result<()> {
    MODEL_CONFIG
        .set(ModelConfig::new(facedet_path, facerec_path))
        .map_err(|_| anyhow::anyhow!("Model config already initialized"))?;
    Ok(())
}
pub fn get_session_pool(model_type: ModelType) -> Result<&'static SessionPool> {
    let path = MODEL_CONFIG
        .get()
        .ok_or_else(|| anyhow::anyhow!("Model config not initialized"))?
        .get_model_path(model_type);

    let pool_size = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(6).max(2);

     let pool = match model_type {
        ModelType::FaceDet => &FACE_DET_POOL,
        ModelType::FaceRec => &FACE_REC_POOL,
    };

    if pool.get().is_none() {
        let new_pool = SessionPool::new(path, pool_size)?;
        let _ = pool.set(new_pool);
    }

    Ok(pool.get().unwrap())
}

/// 全局会话池单例
// static GLOBAL_SESSION_POOL: OnceLock<SessionPool> = OnceLock::new();

// /// 获取全局会话池
// /// 
// /// 返回全局唯一的人脸识别会话池实例，如果不存在则会创建一个新的。
// /// 池大小基于CPU核心数自动确定，最少为2个会话。
// /// 
// /// # 返回值
// /// 
// /// 返回全局会话池的引用
// /// 
// /// # 错误
// /// 
// /// 如果模型加载失败或池创建失败会返回错误
// pub fn get_global_session_pool(module_path: &str) -> Result<&'static SessionPool> {
//     if let Some(pool) = GLOBAL_SESSION_POOL.get() {
//         return Ok(pool);
//     }
    
//     let pool_size = std::thread::available_parallelism()
//         .map(|n| n.get())
//         .unwrap_or(6)
//         .max(2);
    
//     let pool = SessionPool::new(module_path, pool_size)?;
//     GLOBAL_SESSION_POOL.set(pool).map_err(|_| anyhow::anyhow!("Failed to set global session pool"))?;
//     Ok(GLOBAL_SESSION_POOL.get().unwrap())
// }
impl BoundingBox{
    /// 创建新的边界框
    /// 
    /// # 参数
    /// 
    /// * `x1` - 左上角x坐标
    /// * `y1` - 左上角y坐标
    /// * `x2` - 右下角x坐标
    /// * `y2` - 右下角y坐标
    /// 
    /// # 返回值
    /// 
    /// 返回新的BoundingBox实例
    pub fn new(  x1:i32,y1:i32,x2:i32,y2:i32,) -> Self{
        Self{x1,y1,x2,y2}
    }
    
    /// 计算边界框的宽度
    /// 
    /// # 返回值
    /// 
    /// 返回边界框的宽度（绝对值）
    pub fn width(&self) -> f32{
        (self.x2 - self.x1).abs() as f32
    }
    
    /// 计算边界框的高度
    /// 
    /// # 返回值
    /// 
    /// 返回边界框的高度（绝对值）
    pub fn heigth(&self) -> f32{
        (self.y2 - self.y1).abs() as f32
    }
}

/// 人脸检测结果
/// 
/// 包含检测到的人脸边界框、五个关键点、置信度等信息。
/// 
/// # 字段
/// 
/// * `bbox` - 人脸边界框
/// * `fivepoint` - 五个关键点（左眼、右眼、鼻尖、左嘴角、右嘴角）
/// * `conf` - 检测置信度
/// * `class` - 类别标识
#[derive(Debug,Clone)]
pub struct FaceDetection{
    pub bbox:BoundingBox,
    pub fivepoint:Vec<Point>,
    pub conf:f32,
    pub class:i32,
}

impl FaceDetection{
    /// 创建新的人脸检测结果
    /// 
    /// # 参数
    /// 
    /// * `bbox` - 人脸边界框
    /// * `fivepoint` - 五个关键点
    /// * `conf` - 检测置信度
    /// * `class` - 类别标识
    /// 
    /// # 返回值
    /// 
    /// 返回新的FaceDetection实例
    pub fn new(bbox:BoundingBox,fivepoint:Vec<Point>,conf:f32,class:i32)->Self{
        Self {bbox,fivepoint,conf,class}
    }
    
    /// 计算人脸姿态角度
    /// 
    /// 基于五个关键点计算人脸的俯仰角(pitch)、偏航角(yaw)和翻滚角(roll)。
    /// 
    /// # 参数
    /// 
    /// * `_image_width` - 图像宽度（当前未使用）
    /// * `_image_height` - 图像高度（当前未使用）
    /// 
    /// # 返回值
    /// 
    /// 返回FacePoseAngles结构体，包含三个角度值
    pub fn calculate_pose_angles(&self, _image_width: u32, _image_height: u32) -> FacePoseAngles {
        let left_eye = self.fivepoint[0];
        let right_eye = self.fivepoint[1];
        let nose = self.fivepoint[2];
        let mouth_left = self.fivepoint[3];
        let mouth_right = self.fivepoint[4];
        
        let eye_center_y = (left_eye.y + right_eye.y) as f32 / 2.0;
        let mouth_center_y = (mouth_left.y + mouth_right.y) as f32 / 2.0;
        let expected_nose_y = (eye_center_y + mouth_center_y) / 2.0;
        let nose_deviation = nose.y as f32 - expected_nose_y;
        let face_height = (mouth_center_y - eye_center_y).abs();
        let pitch = if face_height > 0.0 {
            (nose_deviation / face_height).clamp(-1.0, 1.0) * 20.0
        } else {
            0.0
        };
        
        let eye_center_x = (left_eye.x + right_eye.x) as f32 / 2.0;
        let eye_distance = (right_eye.x - left_eye.x).abs() as f32;
        let nose_offset = nose.x as f32 - eye_center_x;
        let yaw = if eye_distance > 0.0 {
            (nose_offset / eye_distance).clamp(-1.0, 1.0) * 30.0
        } else {
            0.0
        };
        
        let eye_dx = right_eye.x as f32 - left_eye.x as f32;
        let eye_dy = right_eye.y as f32 - left_eye.y as f32;
        let roll = if eye_dx.abs() > 0.0 {
            (eye_dy / eye_dx).atan() * (180.0 / std::f32::consts::PI)
        } else {
            0.0
        };
        
        FacePoseAngles::new(pitch, yaw, roll)
    }
}
/// 人脸特征向量
/// 
/// 表示从人脸图像中提取的特征向量，用于人脸识别和相似度计算。
/// 
/// # 字段
/// 
/// * `id` - 可选的人脸ID
/// * `vector` - 特征向量（已归一化）
#[derive(Debug,Clone)]
pub struct FaceFeature{
    pub id: Option<u64>,
    pub vector:Vec<f32>
}

impl FaceFeature{
    /// 创建新的人脸特征
    /// 
    /// # 参数
    /// 
    /// * `vector` - 特征向量
    /// 
    /// # 返回值
    /// 
    /// 返回新的FaceFeature实例，ID为None
    pub fn new(vector:Vec<f32>) -> Self{
        Self{id: None, vector}
    }
    
    /// 创建带ID的人脸特征
    /// 
    /// # 参数
    /// 
    /// * `id` - 人脸ID
    /// * `vector` - 特征向量
    /// 
    /// # 返回值
    /// 
    /// 返回新的FaceFeature实例，包含指定的ID
    pub fn new_with_id(id:u64,vector:Vec<f32>) -> Self{
        Self{id: Some(id), vector}
    }
    
    /// 计算与另一个人脸特征的相似度
    /// 
    /// 使用基于欧氏距离的相似度计算公式。
    /// 
    /// # 参数
    /// 
    /// * `other` - 另一个人脸特征
    /// 
    /// # 返回值
    /// 
    /// 返回相似度值，范围在0-1之间，越接近1表示越相似
    pub fn similarity(&self,other:&FaceFeature)->f32{
    let dist_squared: f32 = self.vector.iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        1.0 / (1.0 + (-((1.40 - dist_squared) / 0.2)).exp())
    }
}

impl AsRef<FaceFeature> for FaceFeature {
    fn as_ref(&self) -> &FaceFeature {
        self
    }
}


/// 支持的图像格式
/// 
/// 枚举了库支持的各种图像格式。
#[derive(Debug, Clone, Copy)]
pub enum ImageFormat {
    /// JPEG格式
    Jpeg,
    /// PNG格式
    Png,
    /// WebP格式
    WebP,
    /// BMP格式
    Bmp,
    /// 未知格式
    Unknown,
}

/// 统一的解码图像结构
/// 
/// 包含解码后的图像数据和元信息。
/// 
/// # 字段
/// 
/// * `width` - 图像宽度
/// * `height` - 图像高度
/// * `rgb_data` - RGB格式图像数据
/// * `format` - 图像格式
#[derive(Debug, Clone)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgb_data: Vec<u8>,
    pub format: ImageFormat,
}

impl DecodedImage {
    /// 创建新的解码图像
    /// 
    /// # 参数
    /// 
    /// * `width` - 图像宽度
    /// * `height` - 图像高度
    /// * `rgb_data` - RGB数据
    /// * `format` - 图像格式
    /// 
    /// # 返回值
    /// 
    /// 返回新的DecodedImage实例
    pub fn new(width: u32, height: u32, rgb_data: Vec<u8>, format: ImageFormat) -> Self {
        Self { width, height, rgb_data, format }
    }
    
    /// 根据文件头检测图像格式
    /// 
    /// 通过检查文件头部的魔数来识别图像格式。
    /// 
    /// # 参数
    /// 
    /// * `bytes` - 图像文件字节数据
    /// 
    /// # 返回值
    /// 
    /// 返回检测到的ImageFormat
    pub fn detect_format(bytes: &[u8]) -> ImageFormat {
        if bytes.len() >= 4 {
            match &bytes[0..4] {
                b"\xFF\xD8\xFF" => ImageFormat::Jpeg,
                b"\x89PNG" => ImageFormat::Png,
                b"RIFF" if bytes.len() >= 12 && &bytes[8..12] == b"WEBP" => ImageFormat::WebP,
                b"BM" => ImageFormat::Bmp,
                _ => ImageFormat::Unknown,
            }
        } else {
            ImageFormat::Unknown
        }
    }
}

/// 车牌识别结果
/// 
/// 包含车牌识别的相关信息。
/// 
/// # 字段
/// 
/// * `plate_no` - 车牌号码
/// * `plate_color` - 车牌颜色
/// * `roi_height` - 感兴趣区域高度
/// * `landmarks` - 关键点
#[derive(Debug,Clone)]
pub struct PlateResult {
    pub plate_no: String,
    pub plate_color: String,
    pub roi_height: u32,
    pub landmarks: Vec<Point>,
}

impl PlateResult {
    /// 创建新的车牌识别结果
    /// 
    /// # 参数
    /// 
    /// * `plate_no` - 车牌号码
    /// * `plate_color` - 车牌颜色
    /// * `roi_height` - 感兴趣区域高度
    /// * `landmarks` - 关键点
    /// 
    /// # 返回值
    /// 
    /// 返回新的PlateResult实例
    pub fn new(plate_no: String, plate_color: String, roi_height: u32, landmarks: Vec<Point>) -> Self {
        Self { plate_no, plate_color, roi_height, landmarks }
    }
}
