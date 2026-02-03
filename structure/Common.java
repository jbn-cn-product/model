package com.example.model.structure;

public class Common {

    // 坐标
    public static class Point {
        public float x;
        public float y;
        public Point() {}
        public Point(float x, float y) {
            this.x = x;
            this.y = y;
        }
    }

    // 检测框
    public static class Box {
        public Point point; // 左上角原点
        public float width;
        public float height;
        public Box() {}
        public Box(Point point, float width, float height) {
            this.point = point;
            this.width = width;
            this.height = height;
        }
    }

}
