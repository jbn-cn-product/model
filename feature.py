import numpy as np
import cv2
import onnxruntime as ort
import os
from face_detector import FaceDetector
import time

class Feature:
    def __init__(self, model_path="models/face_rec.onnx"):

        self.model_path = model_path
        
        # ONNX推理配置
        providers = ['CPUExecutionProvider']

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"Failed to load feature model: {e}")
            raise

    def preprocess(self, face_img):
        """预处理人脸图片"""
        # BGR转RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # 直接使用112x112，不进行resize和crop
        if face_img.shape[:2] != (112, 112):
            face_rgb = cv2.resize(face_rgb, (112, 112))
        
        # 归一化到[0,1] - 确保使用float32
        # face_normalized = face_rgb.astype(np.float32) / 255.0
        # ImageNet标准归一化 - 确保所有参数都是float32

        face_normalized = (face_rgb - 127.5) / 127.5
        # 转换维度 [H,W,C] -> [1,C,H,W]
        face_transposed = face_normalized.transpose(2, 0, 1)
        face_batch = np.expand_dims(face_transposed, axis=0).astype(np.float32)
        return face_batch

    def postprocess(self, embeddings):
        """后处理：L2归一化"""
        # 确保输入是float32
        embeddings = embeddings.astype(np.float32)
        
        # L2归一化
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norm + 1e-8)
        
        return normalized_embeddings.astype(np.float32)

    def get_reference_points(self):
        """获取112x112的参考关键点坐标"""
        # 原始96x112的参考点
        kpts_ref = np.array([
            [30.29459953, 51.69630051],  # 左眼
            [65.53179932, 51.50139999],  # 右眼  
            [48.02519989, 71.73660278],  # 鼻尖
            [33.54930115, 92.3655014],   # 左嘴角
            [62.72990036, 92.20410156]   # 右嘴角
        ], dtype=np.float32)
        
        # 转换为112x112格式（居中）
        kpts_ref[:, 0] += 8.0  # x方向偏移 (112-96)/2 = 8
        return kpts_ref


    def face_align(self, image, landmarks):
        """人脸对齐函数：先仿射变换，直接输出112x112对齐人脸"""
        # 获取参考关键点
        kpts_ref = self.get_reference_points()
        # 转换输入关键点格式
        kpts = np.array(landmarks, dtype=np.float32)
        # 计算仿射变换矩阵
        transform_matrix, _ = cv2.estimateAffine2D(kpts, kpts_ref)
        if transform_matrix is None:
            # 如果estimateAffine2D失败，使用前3个点的仿射变换
            transform_matrix = cv2.getAffineTransform(kpts[:3], kpts_ref[:3])
        
        # 直接应用仿射变换到原图，输出112x112对齐人脸
        aligned_face = cv2.warpAffine(image, transform_matrix, (112, 112))
        
        return aligned_face

    def feature(self, img, landmarks):
        """特征提取主函数"""
        face_img = self.face_align(img, landmarks)
        # 前处理
        input_tensor = self.preprocess(face_img)
        # ONNX推理
        outputs = self.session.run(None, {'input.1': input_tensor})
        embeddings = outputs[0]  # [1, 512]
        # 后处理
        normalized_embeddings = self.postprocess(embeddings)
        return normalized_embeddings

    @staticmethod
    def euclidean_distance(feat1, feat2):
        """计算欧氏距离"""
        distance = np.linalg.norm(feat1 - feat2)
        return float(distance)

    @staticmethod
    def calculate_similarity(feat1, feat2):
        """计算两个归一化向量的相似度"""
        feat1 = feat1.flatten()
        feat2 = feat2.flatten()
        
        euclidean_dist = np.linalg.norm(feat1 - feat2)
        
        dist_squared = np.sum((feat1 - feat2) ** 2)
        sigmoid_score = 1.0 / (1.0 + np.exp(-((1.40 - dist_squared) / 0.2)))
        
        return {'sigmoid_score': sigmoid_score}   # [0, 1], 越大越相似
