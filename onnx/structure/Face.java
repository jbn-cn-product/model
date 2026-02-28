package com.example.model.onnx.structure;

import com.example.model.onnx.structure.Position.Point;

public class Face {

    public Point leftEye;       // 左眼
    public Point rightEye;      // 右眼
    public Point nose;          // 鼻子
    public Point leftMouth;     // 左嘴角
    public Point rightMouth;    // 右嘴角

    public float yaw = 0.0f;    // 偏航角
    public float pitch = 0.0f;  // 俯仰角
    public float roll = 0.0f;   // 翻滚角

    public Face(Point leftEye, Point rightEye, Point nose, Point leftMouth, Point rightMouth) {
        this.leftEye = leftEye;
        this.rightEye = rightEye;
        this.nose = nose;
        this.leftMouth = leftMouth;
        this.rightMouth = rightMouth;
        // 计算人脸姿态
        float eyeCenterX = (leftEye.x + rightEye.x) / 2.0f;
        float eyeDistance = Math.abs(rightEye.x - leftEye.x);
        float noseOffset = nose.x - eyeCenterX;
        if (eyeDistance > 0.0f) {
            yaw = Math.max(-1.0f, Math.min(1.0f, noseOffset / eyeDistance)) * 30.0f;
        }
        float eyeCenterY = (leftEye.y + rightEye.y) / 2.0f;
        float mouthCenterY = (leftMouth.y + rightMouth.y) / 2.0f;
        float noseDeviation = nose.y - ((eyeCenterY + mouthCenterY) / 2.0f);
        float faceHeight = Math.abs(mouthCenterY - eyeCenterY);
        if (faceHeight > 0.0f) {
            pitch = Math.max(-1.0f, Math.min(1.0f, noseDeviation / faceHeight)) * 20.0f;
        }
        float eyeDx = rightEye.x - leftEye.x;
        float eyeDy = rightEye.y - leftEye.y;
        if (Math.abs(eyeDx) > 0.0f) {
            roll = (float) Math.toDegrees(Math.atan(eyeDy / eyeDx));
        }
    }

}
