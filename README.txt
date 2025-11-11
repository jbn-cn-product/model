# 项目说明

本项目实现了车牌和人脸的检测、识别与特征提取，基于 ONNX 模型，支持 CPU 推理。

## 功能
- **人脸**：检测 + 对齐 + 提取特征向量，可用于相似度匹配。
- **车牌**：检测 + 透视矫正 + 识别号码和颜色。

安装依赖：
   pip install opencv-python numpy onnxruntime

准备模型到 `models/` 文件夹：
   - face_rec.onnx         特征提取（人脸）
   - car_face_det.onnx    检测模型
   - car_rec.onnx     车牌识别

运行示例：
   python face_plate.py
   - 输出检测结果、人脸特征向量或车牌识别结果。
