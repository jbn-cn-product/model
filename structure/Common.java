package com.example.model.structure;

public class Common {

    // 坐标
    public static class Point {
        public int x;
        public int y;
        public Point() {}
        public Point(int x, int y) {
            this.x = x;
            this.y = y;
        }
    }

    // 检测框
    public static class Box {
        public Point point; // 左上角原点
        public int width;
        public int height;
        public Box() {}
        public Box(Point point, int width, int height) {
            this.point = point;
            this.width = width;
            this.height = height;
        }
    }

}
