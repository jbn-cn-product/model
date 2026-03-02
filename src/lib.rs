mod rknn_ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(unused_imports)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub mod utils;
pub mod yolo;

pub enum TensorData {
    Float(Vec<f32>),
    UInt8(Vec<u8>),
}

pub trait Context {
    fn pre(&self, image_bytes: &[u8]) -> Result<(TensorData, LetterBoxInfo), String>;
    fn inference(&self, preprocessed_data: &TensorData) -> Result<Vec<Vec<f32>>, String>;
}

#[derive(Debug, Clone)]
pub struct Object {
    pub bbox: [f32; 4],
    pub confidence: f32,
    pub class_id: i32,
}

#[derive(Debug, Clone)]
pub struct LetterBoxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub unpad_w: i32,
    pub unpad_h: i32,
    pub org_w: i32,
    pub org_h: i32,
}
