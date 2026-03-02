import cv2
import numpy as np
import onnxruntime
from itertools import product
from math import ceil

class FaceDetector:
    def __init__(self, model_path="models/face_det.onnx"):
        """初始化RetinaFace ONNX检测器"""
        self.model_path = model_path
        self.min_face_confidence = 0.95
        # RetinaFace配置
        self.cfg_mnet = {
            'min_sizes': [[16, 32], [64, 128], [256, 512]],
            'steps': [8, 16, 32],
            'variance': [0.1, 0.2],}
        
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Failed to load RetinaFace model: {e}")
            raise
    
    def letterbox_image(self, image, size):
        """图像预处理：letterbox缩放"""
        ih, iw = image.shape[:2]
        w, h = size
        scale = min(w/iw, h/ih)
        nw, nh = int(iw*scale), int(ih*scale)
        image = cv2.resize(image, (nw, nh))
        new_image = np.ones([h, w, 3], dtype=np.uint8) * 128
        new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
        return new_image, scale, (w-nw)//2, (h-nh)//2
    
    def get_anchors(self, image_size=(640, 640)):
        """生成anchor"""
        feature_maps = [[ceil(image_size[0]/step), ceil(image_size[1]/step)] for step in self.cfg_mnet['steps']]
        anchors = []
        for k, f in enumerate(feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in self.cfg_mnet['min_sizes'][k]:
                    cx = (j + 0.5) * self.cfg_mnet['steps'][k] / image_size[1]
                    cy = (i + 0.5) * self.cfg_mnet['steps'][k] / image_size[0]
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    anchors += [cx, cy, s_kx, s_ky]
        return np.array(anchors).reshape(-1, 4)
    
    def decode(self, loc, priors, variances):
        """解码bbox"""
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
    
    def decode_landm(self, pre, priors, variances):
        """解码landmarks"""
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                            ), 1)
        return landms
    
    def nms(self, dets, thresh):
        """非极大值抑制"""
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (y2 - y1) * (x2 - x1)
        keep = []
        index = scores.argsort()[::-1]
        
        while index.size > 0:
            i = index[0]
            keep.append(i)
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])
            
            w = np.maximum(0, x22 - x11)
            h = np.maximum(0, y22 - y11)
            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep
    
    def detect(self, image):
        if image is None:
            return []
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        
        # 预处理
        resized_img, scale, pad_w, pad_h = self.letterbox_image(image, (640, 640))
        
        img = resized_img.astype(np.float32)
        img -= np.array((104, 117, 123), np.float32)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        
        # 推理
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})
        
        # 解码
        anchors = self.get_anchors((640, 640))
        boxes = self.decode(outputs[0].squeeze(), anchors, self.cfg_mnet['variance'])
        landms = self.decode_landm(outputs[2].squeeze(), anchors, self.cfg_mnet['variance'])
        conf = outputs[1].squeeze()[:, 1:2]
        
        # 组合并过滤
        dets = np.concatenate((boxes, conf, landms), axis=1)
        valid_idx = dets[:, 4] > self.min_face_confidence
        dets = dets[valid_idx]
        
        if len(dets) == 0:
            return []
        
        # NMS
        keep = self.nms(dets, 0.6)
        dets = dets[keep]
        
        # 转换回原图坐标
        bboxes = dets[:, :4] * 640
        landmarks = dets[:, 5:] * 640
        scores = dets[:, 4]
        
        # 去除letterbox填充的影响
        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_w) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_h) / scale
        landmarks[:, 0::2] = (landmarks[:, 0::2] - pad_w) / scale
        landmarks[:, 1::2] = (landmarks[:, 1::2] - pad_h) / scale
        
        # 格式化输出结果，应用过滤逻辑
        results = []
        for i in range(len(bboxes)):
            bbox = bboxes[i].astype(int).tolist()  # [x1, y1, x2, y2]
            confidence = float(scores[i])
            
            # 边界校正
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(width, bbox[2])
            bbox[3] = min(height, bbox[3])
            
            # 转换关键点格式：[x1,y1,x2,y2,...,x10,y10] -> [[x1,y1],[x2,y2],...,[x5,y5]]
            landmark_points = []
            out_edge = False
            landmark_raw = landmarks[i]
            for j in range(5):
                x_coord = float(landmark_raw[j*2])
                y_coord = float(landmark_raw[j*2+1])
                if x_coord < 0 or y_coord < 0 or x_coord > width or y_coord > height:
                    out_edge = True
                landmark_points.append([x_coord, y_coord])
            
            if out_edge:
                continue
            
            results.append({
                'bbox': bbox,
                'landmarks': landmark_points,
                'confidence': confidence
            })
        return results
