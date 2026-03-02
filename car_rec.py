import onnxruntime
import numpy as np
import cv2
from car_detector import CarDetector
from PIL import Image, ImageDraw, ImageFont
import os

class PlateRecognizer:
    def __init__(self, rec_model_path="models/car_rec.onnx", providers=['CPUExecutionProvider']):

        self.rec_model_path = rec_model_path
        self.providers = providers
        self.session_rec = onnxruntime.InferenceSession(rec_model_path, providers=providers)
        
        self.plate_color_list = ['黑色','蓝色','绿色','白色','黄色']
        self.plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
        self.mean_value, self.std_value = 0.588, 0.193

    def order_points(self, pts):     # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):  #透视变换得到矫正后的图像，方便识别
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
        # return the warped image
        return warped

    def rec_pre_processing(self, img):  # 识别前处理
        img = cv2.resize(img, (168, 48))
        img = img.astype(np.float32)
        img = (img/255-self.mean_value)/self.std_value  #归一化 减均值 除标准差
        img = img.transpose(2,0,1)         #h,w,c 转为 c,h,w
        img = img.reshape(1,*img.shape)
        return img

    def decodePlate(self, preds):        #识别后处理
        pre=0
        newPreds=[]
        for i in range(len(preds)):
            if preds[i]!=0 and preds[i]!=pre:
                newPreds.append(preds[i])
            pre=preds[i]
        plate=""
        for i in newPreds:
            plate+=self.plateName[int(i)]
        return plate

    def get_plate_result(self, img): #识别后处理
        img = self.rec_pre_processing(img)
        y_onnx_plate,y_onnx_color = self.session_rec.run([self.session_rec.get_outputs()[0].name,self.session_rec.get_outputs()[1].name], {self.session_rec.get_inputs()[0].name: img})
        index =np.argmax(y_onnx_plate,axis=-1)
        index_color = np.argmax(y_onnx_color)
        plate_color = self.plate_color_list[index_color]
        plate_no = self.decodePlate(index[0])
        return plate_no,plate_color

    def get_split_merge(self, img):  #双层车牌进行分割后识别
        h,w,c = img.shape
        img_upper = img[0:int(5/12*h),:]
        img_lower = img[int(1/3*h):,:]
        img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
        new_img = np.hstack((img_upper,img_lower))
        return new_img

    def rec(self, img, land_marks, label):
        """车牌识别"""
        land_marks = np.array(land_marks, dtype=np.float32)
        result={}
        roi_img = self.four_point_transform(img,land_marks)
        if label==1:  #双层车牌
            roi_img = self.get_split_merge(roi_img)
        plate_no,plate_color = self.get_plate_result(roi_img)
        result['plate_no']=plate_no
        result['roi_height']=roi_img.shape[0]
        result['plate_color']=plate_color
        result['landmarks']=self.order_points(land_marks).astype(int).tolist()
        return result

def draw_plate_result(img, bbox, rec_result):
    # 绘制检测框
    (x1, y1, x2, y2) = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    land_marks = rec_result['landmarks']
    # 绘制关键点
    for (x, y) in land_marks:
        cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

    # 准备绘制中文文字
    text = f"{rec_result['plate_no']} ({rec_result['plate_color']})"

    # 转成 PIL 格式绘制中文
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 加载字体（确保路径正确）
    font_path = "simhei.ttf"
    font = ImageFont.truetype(font_path, 28)

    # 计算文字位置（避免出界）
    text_x = max(0, x1)
    text_y = max(0, y1 - 35)
    draw.text((text_x, text_y), text, font=font, fill=(255, 0, 0))

    # 转回 OpenCV 格式
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img