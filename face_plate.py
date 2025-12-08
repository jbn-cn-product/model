import cv2
import numpy as np
import onnxruntime
import copy
import time
import os

class Feature:
    def __init__(self, model_path="models/face_rec.onnx"):
        self.model_path = model_path
        providers = ['CPUExecutionProvider']
        try:
            if os.path.exists(model_path):
                self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            else:
                print(f"[Face] Model not found: {model_path} (Skipping feature extraction)")
                self.session = None
        except Exception as e:
            print(f"[Face] Failed to load model: {e}")
            self.session = None

    def preprocess(self, face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        if face_img.shape[:2] != (112, 112):
            face_rgb = cv2.resize(face_rgb, (112, 112))
        face_normalized = (face_rgb - 127.5) / 127.5
        face_transposed = face_normalized.transpose(2, 0, 1)
        face_batch = np.expand_dims(face_transposed, axis=0).astype(np.float32)
        return face_batch

    def postprocess(self, embeddings):
        embeddings = embeddings.astype(np.float32)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norm + 1e-8)
        return normalized_embeddings.astype(np.float32)

    def get_reference_points(self):
        kpts_ref = np.array([
            [30.29459953, 51.69630051],
            [65.53179932, 51.50139999],
            [48.02519989, 71.73660278],
            [33.54930115, 92.3655014],
            [62.72990036, 92.20410156]
        ], dtype=np.float32)
        kpts_ref[:, 0] += 8.0
        return kpts_ref

    def face_align(self, image, landmarks):
        kpts_ref = self.get_reference_points()
        kpts = np.array(landmarks, dtype=np.float32)
        if kpts.shape[0] > 5: kpts = kpts[:5]
        
        transform_matrix, _ = cv2.estimateAffine2D(kpts, kpts_ref)
        if transform_matrix is None:
            transform_matrix = cv2.getAffineTransform(kpts[:3], kpts_ref[:3])
        aligned_face = cv2.warpAffine(image, transform_matrix, (112, 112))
        return aligned_face

    def feature(self, img, landmarks):
        if self.session is None: return np.zeros((1, 512))
        face_img = self.face_align(img, landmarks)
        input_tensor = self.preprocess(face_img)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        return self.postprocess(outputs[0])

class Detector:
    def __init__(self, model_path="models/3dbest.onnx"):
        self.model_path = model_path
        self.img_size = (640, 640)
        try:
            if os.path.exists(model_path):
                self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                print(f"[Detector] Loaded: {model_path}")
            else:
                print(f"[Detector] File not found: {model_path}")
                self.session = None
        except Exception as e:
            print(f"[Detector] Error: {e}")
            raise

    def xywh2xyxy(self, boxes):
        xywh = boxes.copy()
        xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return xywh

    def box_iou(self, box1, box2):
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
        inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
        inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
        inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        union = area1[:, None] + area2 - inter
        return inter / (union + 1e-6)

    def nms_numpy(self, boxes, scores, iou_thres=0.45):
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1: break
            ious = self.box_iou(boxes[i:i+1], boxes[order[1:]])[0]
            inds = np.where(ious <= iou_thres)[0]
            order = order[inds + 1]
        return keep

    def non_max_suppression_face(self, prediction, conf_thres=0.25, iou_thres=0.45):
        output = []
        for img_pred in prediction:
            x = img_pred[img_pred[:, 4] > conf_thres]
            if not x.shape[0]:
                output.append(np.zeros((0, 16)))
                continue
            x[:, 15:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            conf = np.max(x[:, 15:], axis=1)
            j = np.argmax(x[:, 15:], axis=1)
            x = np.concatenate((box, conf[:, None], x[:, 5:15], j[:, None]), axis=1)
            x = x[conf > conf_thres]
            if not len(x):
                output.append(np.zeros((0, 16)))
                continue
            boxes, scores = x[:, :4], x[:, 4]
            keep = self.nms_numpy(boxes, scores, iou_thres)
            output.append(x[keep])
        return output

    def my_letter_box(self, img, size=(640, 640)):
        h, w, c = img.shape
        r = min(size[0] / h, size[1] / w)
        new_h, new_w = int(h * r), int(w * r)
        top = int((size[0] - new_h) / 2)
        left = int((size[1] - new_w) / 2)
        bottom = size[0] - new_h - top
        right = size[1] - new_w - left
        img_resize = cv2.resize(img, (new_w, new_h))
        img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, left, top

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]; pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])
        coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])
        coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])
        coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])
        return coords

    def scale_coords_landmarks(self, img1_shape, coords, img0_shape, ratio_pad=None):
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]; pad = ratio_pad[1]
        coords[:, [0, 2, 4, 6, 8]] -= pad[0]
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]
        coords[:, :10] /= gain
        coords[:, 0::2] = np.clip(coords[:, 0::2], 0, img0_shape[1])
        coords[:, 1::2] = np.clip(coords[:, 1::2], 0, img0_shape[0])
        return coords

    def detect(self, img):
        if self.session is None: return []
        img0 = copy.deepcopy(img)
        img_resized, r, left, top = self.my_letter_box(img0, size=self.img_size)
        im = img_resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        im /= 255.0
        im = np.expand_dims(im, axis=0)
        
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        pred = self.session.run([output_name], {input_name: im})[0]
        preds = self.non_max_suppression_face(pred, 0.3, 0.5)
        
        results = []
        for det in preds:
            if len(det):
                det[:, :4] = self.scale_coords(im.shape[2:], det[:, :4], img0.shape)
                det[:, 5:15] = self.scale_coords_landmarks(im.shape[2:], det[:, 5:15], img0.shape)
                for j in range(det.shape[0]):
                    xyxy = det[j, :4]
                    conf = det[j, 4]
                    landmarks = det[j, 5:15]
                    cls = det[j, 15]
                    results.append({
                        'cls': int(cls),
                        'bbox': list(map(int, xyxy.tolist())),
                        'landmarks': [(int(landmarks[k]), int(landmarks[k+1])) for k in range(0, len(landmarks), 2)],
                        'confidence': conf
                    })
        return results

class PlateRecognizer:
    def __init__(self, rec_model_path="models/car_rec.onnx", providers=['CPUExecutionProvider']):
        if os.path.exists(rec_model_path):
            self.session_rec = onnxruntime.InferenceSession(rec_model_path, providers=providers)
        else:
            print(f"[Plate] Model not found: {rec_model_path}")
            self.session_rec = None
        self.plate_color_list = ['黑色','蓝色','绿色','白色','黄色']
        self.plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        self.mean_value, self.std_value = 0.588, 0.193

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def rec_pre_processing(self, img):
        img = cv2.resize(img, (168, 48))
        img = img.astype(np.float32)
        img = (img/255-self.mean_value)/self.std_value
        img = img.transpose(2,0,1)
        img = img.reshape(1,*img.shape)
        return img

    def decodePlate(self, preds):
        pre=0
        newPreds=[]
        for i in range(len(preds)):
            if preds[i]!=0 and preds[i]!=pre: newPreds.append(preds[i])
            pre=preds[i]
        plate=""
        for i in newPreds: plate+=self.plateName[int(i)]
        return plate

    def get_plate_result(self, img):
        if self.session_rec is None: return "Unknown", "Unknown"
        img = self.rec_pre_processing(img)
        outputs = self.session_rec.run(None, {self.session_rec.get_inputs()[0].name: img})
        y_onnx_plate, y_onnx_color = outputs[0], outputs[1]
        plate_no = self.decodePlate(np.argmax(y_onnx_plate,axis=-1)[0])
        plate_color = self.plate_color_list[np.argmax(y_onnx_color)]
        return plate_no, plate_color

    def rec(self, img, land_marks):
        pts = np.array(land_marks, dtype=np.float32)
        if pts.shape[0] > 4: pts = pts[:4, :] # 只取前4个角点
        
        roi_img = self.four_point_transform(img, pts)
        plate_no, plate_color = self.get_plate_result(roi_img)
        return {'plate_no': plate_no, 'plate_color': plate_color, 'roi_height': roi_img.shape[0]}

class LicenseHandler:
    def __init__(self):
        # 无需模型
        pass
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)] # TL
        rect[2] = pts[np.argmax(s)] # BR
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        return rect

    # 复用透视变换逻辑
    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        # 计算新宽高
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # 变换
        dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def process(self, img, landmarks):
        """执照处理主函数"""
        pts = np.array(landmarks, dtype=np.float32)
        if pts.shape[0] > 4:
            pts = pts[:4, :]
        
        # 抠图并拉直
        roi_img = self.four_point_transform(img, pts)
        return roi_img

def main():
    start_time = time.time()
    det = Detector(model_path="/home/server/ZhangTonghua/car_face_inone/models/3dbest.onnx") 
    car_rec = PlateRecognizer(rec_model_path="models/car_rec.onnx") 
    face_feature = Feature(model_path="models/face_rec.onnx")

    license_handler = LicenseHandler()
    
    print(f"Models loaded in {time.time() - start_time:.2f}s")

    # --- B. 读取图片 ---
    img_path = "/home/server/ZhangTonghua/car_face_inone/test_img/face1.jpg"  # 修改为你的图片路径
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    print(f"Processing: {img_path}")
    detect_results = det.detect(img)
    print(f"Found {len(detect_results)} objects.")

    # --- D. 遍历结果并分发处理 ---
    for i, res in enumerate(detect_results):
        cls = res['cls']
        bbox = res['bbox']
        landmarks = res['landmarks']
        conf = res['confidence']
        
        print(f"\n--- Object {i+1} [Class {cls}] ---")
        if cls == 0:
            print("Type: Car Plate")
            # 传入原图和关键点，进行识别
            rec_res = car_rec.rec(img, landmarks)
            print(f"  > Plate No: {rec_res['plate_no']}")
            print(f"  > Color: {rec_res['plate_color']}")
            
            # 画绿框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            text = f"{rec_res['plate_no']}"
            cv2.putText(img, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif cls == 1:
            print("Type: Face")
            # 传入原图和关键点，提取特征
            feat = face_feature.feature(img, landmarks)
            vec_str = ", ".join([f"{x:.4f}" for x in feat[0][:3]])
            print(f"  > Feature vector: [{vec_str}, ... {feat[0][-1]:.4f}]")
            print(f"  > Feature vector shape: {feat.shape}")
            
            # 画红框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(img, "Face", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif cls == 2:
            print("Type: Business License")
            rectified_img = license_handler.process(img, landmarks)
            # 画蓝框
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(img, "License", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite("final_result.jpg", img)
if __name__ == "__main__":
    main()