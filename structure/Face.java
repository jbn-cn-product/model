package com.example.model.structure;

import com.example.model.structure.Common.Point;

public class Face {

    // 姿态
    public static class Angles {
        public float yaw;   // 偏航角
        public float pitch; // 俯仰角
        public float roll;  // 翻滚角
        public Angles(float yaw, float pitch, float roll) {
            this.yaw = yaw;
            this.pitch = pitch;
            this.roll = roll;
        }
    }

    // 关键点
    public static class Landmarks {
        public Point leftEye;       // 左眼
        public Point rightEye;      // 右眼
        public Point nose;          // 鼻子
        public Point leftMouth;     // 左嘴角
        public Point rightMouth;    // 右嘴角
        public Landmarks(Point leftEye, Point rightEye, Point nose, Point leftMouth, Point rightMouth) {
            this.leftEye = leftEye;
            this.rightEye = rightEye;
            this.nose = nose;
            this.leftMouth = leftMouth;
            this.rightMouth = rightMouth;
        }
    }

}
