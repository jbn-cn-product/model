import cv2
import os
from face_detector import FaceDetector
from car_detector import CarDetector
from car_rec import PlateRecognizer
from feature import Feature
import numpy as np

face_det = FaceDetector()
car_det = CarDetector()
car_rec = PlateRecognizer()
face_feature_extractor = Feature()

input_dir = "test_data"  # 输入文件夹路径
output_dir = "results" # 输出文件夹路径

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历文件夹中的图片
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

print(f"Found {len(image_files)} images in {input_dir}")

for filename in image_files:
    img_path = os.path.join(input_dir, filename)
    print(f"Processing image: {img_path}")
    
    # 使用 imdecode 读取包含中文路径的图片
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    if img is None:
        print(f"Error: Failed to load image from {img_path}")
        continue

    # 如果是4通道图片(RGBA)，转换为3通道(BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    face_det_ress = face_det.detect(img)
    car_det_ress = car_det.detect(img)
    
    if len(face_det_ress)!=0:
        print(f"Detected {len(face_det_ress)} faces.", end=" ")
    for res in face_det_ress:
        bbox = res['bbox']
        land_marks = res['landmarks']
        feature_vector = face_feature_extractor.feature(img, land_marks)

        print(f"Face feature extracted. Confidence: {res['confidence']:.4f}")
        
        # 计算人脸清晰度 (拉普拉斯方差)
        # 1. 裁剪人脸区域
        x1, y1, x2, y2 = bbox
        face_roi = img[y1:y2, x1:x2]
        if face_roi.size > 0:
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            print(f"Face Blur Score: {blur_score:.2f} (越低越模糊，通常 < 100 算模糊)")
        
        # 绘制人脸框和关键点
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for (x, y) in land_marks:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), -1)

    if len(car_det_ress)!=0:
        print(f"Detected {len(car_det_ress)} car plates.", end=" ")
    for res in car_det_ress:
        cls = res['cls']
        bbox = res['bbox']
        land_marks = res['landmarks']
        rec_result = car_rec.rec(img, land_marks, cls)
        print(f"  车牌号: {rec_result['plate_no']}, 颜色: {rec_result['plate_color']}\n")
        
        # 绘制车牌框 (只画框，不写字)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存结果图片
    output_path = os.path.join(output_dir, f"result_{filename}")
    # 支持中文路径保存
    cv2.imencode('.jpg', img)[1].tofile(output_path)
    print(f"Result saved to: {output_path}\n")
