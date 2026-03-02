import cv2
import numpy as np
import onnxruntime
import copy

class CarDetector:
    def __init__(self, model_path="models/car_det.onnx"):
        self.model_path = model_path
        self.img_size = (640, 640)
        
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Failed to load RetinaFace model: {e}")
            raise
    
    def my_letter_box(self,img,size=(640,640)):  #
        h,w,c = img.shape
        r = min(size[0]/h,size[1]/w)
        new_h,new_w = int(h*r),int(w*r)
        top = int((size[0]-new_h)/2)
        left = int((size[1]-new_w)/2)
        
        bottom = size[0]-new_h-top
        right = size[1]-new_w-left
        img_resize = cv2.resize(img,(new_w,new_h))
        img = cv2.copyMakeBorder(img_resize,top,bottom,left,right,borderType=cv2.BORDER_CONSTANT,value=(114,114,114))
        return img,r,left,top

    def detect_pre_precessing(self, image, size):
        img,r,left,top=self.my_letter_box(image,size)
        # cv2.imwrite("1.jpg",img)
        img =img[:,:,::-1].transpose(2,0,1).copy().astype(np.float32)
        img=img/255
        img=img.reshape(1,*img.shape)
        return img,r,left,top

    def xywh2xyxy(self, boxes):   #xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
        xywh =copy.deepcopy(boxes)
        xywh[:,0]=boxes[:,0]-boxes[:,2]/2
        xywh[:,1]=boxes[:,1]-boxes[:,3]/2
        xywh[:,2]=boxes[:,0]+boxes[:,2]/2
        xywh[:,3]=boxes[:,1]+boxes[:,3]/2
        return xywh

    def my_nms(self,boxes,iou_thresh):         #nms
        index = np.argsort(boxes[:,4])[::-1]
        keep = []
        while index.size >0:
            i = index[0]
            keep.append(i)
            x1=np.maximum(boxes[i,0],boxes[index[1:],0])
            y1=np.maximum(boxes[i,1],boxes[index[1:],1])
            x2=np.minimum(boxes[i,2],boxes[index[1:],2])
            y2=np.minimum(boxes[i,3],boxes[index[1:],3])
            
            w = np.maximum(0,x2-x1)
            h = np.maximum(0,y2-y1)

            inter_area = w*h
            union_area = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])+(boxes[index[1:],2]-boxes[index[1:],0])*(boxes[index[1:],3]-boxes[index[1:],1])
            iou = inter_area/(union_area-inter_area)
            idx = np.where(iou<=iou_thresh)[0]
            index = index[idx+1]
        return keep

    def restore_box(self,boxes,r,left,top):  #返回原图上面的坐标
        boxes[:,[0,2,5,7,9,11]]-=left
        boxes[:,[1,3,6,8,10,12]]-=top

        boxes[:,[0,2,5,7,9,11]]/=r
        boxes[:,[1,3,6,8,10,12]]/=r
        return boxes


    def post_precessing(self, dets,r,left,top,conf_thresh=0.3,iou_thresh=0.5):#检测后处理
        choice = dets[:,:,4]>conf_thresh
        dets=dets[choice]
        dets[:,13:15]*=dets[:,4:5]
        box = dets[:,:4]
        boxes = self.xywh2xyxy(box)
        score= np.max(dets[:,13:15],axis=-1,keepdims=True)
        index = np.argmax(dets[:,13:15],axis=-1).reshape(-1,1)
        output = np.concatenate((boxes,score,dets[:,5:13],index),axis=1) 
        reserve_=self.my_nms(output,iou_thresh) 
        output=output[reserve_]
        output = self.restore_box(output,r,left,top)
        return output

    def detect(self, img):
        
        img0 = copy.deepcopy(img)
        img,r,left,top = self.detect_pre_precessing(img,self.img_size) #检测前处理
        y_onnx = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: img})[0]
        outputs = self.post_precessing(y_onnx,r,left,top) #检测后处理
        results = []
        for j in range(len(outputs)):
            det = outputs[j]
            xyxy = det[:4]
            conf = det[4]
            landmarks = det[5:13]
            cls = det[13]

            results.append({
                'cls': int(cls),
                'bbox': list(map(int, xyxy.tolist())),
                'landmarks': [(int(landmarks[k]), int(landmarks[k+1])) for k in range(0, len(landmarks), 2)],
                'confidence': conf
            })
        
        return results