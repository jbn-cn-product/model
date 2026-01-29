package com.example.model.structure;

public class Face {

    // 姿态
    public static class Angles {
        public float yaw;   // 偏航角
        public float pitch; // 俯仰角
        public float roll;  // 翻滚角
        public Angles() {}
        public Angles(float yaw, float pitch, float roll) {
            this.yaw = yaw;
            this.pitch = pitch;
            this.roll = roll;
        }
    }

    // 关键点
    public static class Landmarks {
        public int[] leftEye;       // 左眼
        public int[] rightEye;      // 右眼
        public int[] nose;          // 鼻子
        public int[] leftMouth;     // 左嘴角
        public int[] rightMouth;    // 右嘴角
        public Landmarks() {}
        public Landmarks(int[] leftEye, int[] rightEye, int[] nose, int[] leftMouth, int[] rightMouth) {
            this.leftEye = leftEye;
            this.rightEye = rightEye;
            this.nose = nose;
            this.leftMouth = leftMouth;
            this.rightMouth = rightMouth;
        }
    }

}
