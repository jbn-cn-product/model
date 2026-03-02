package com.example.model.structure;

public class Position {

    // 坐标
    public static class Point {
        public float x;
        public float y;
        public Point(float x, float y) {
            this.x = x;
            this.y = y;
        }
        public void restore(Restore restore) {
            x = (x - restore.offsetX) / restore.scale;
            y = (y - restore.offsetY) / restore.scale;
        }
        public boolean isOutOfImage(int originalWidth, int originalHeight) {
            return (x < 0 || y < 0 || y > originalWidth || y > originalHeight);
        }
    }

    // 检测框
    public static class Box {
        public Point point; // 左上角
        public float width;
        public float height;
        public Box(Point point, float width, float height) {
            this.point = point;
            this.width = width;
            this.height = height;
        }
        public void restore(Restore restore) {
            point.restore(restore);
            width /= restore.scale;
            height /= restore.scale;
        }
        public boolean isOutOfImage(int originalWidth, int originalHeight) {
            return (point.x < 0 || point.y < 0 || point.x + width > originalWidth || point.y + height > originalHeight);
        }
    }

    // 图像变换信息
    public static class Restore {
        public float scale;
        public float offsetX;
        public float offsetY;
        public Restore(int imageWidth, int imageHeight, int modelWidth, int modelHeight) {
            scale = Math.min((float) modelWidth / imageWidth, (float) modelHeight / imageHeight);
            offsetX = (modelWidth - imageWidth * scale) / 2.0f;
            offsetY = (modelHeight - imageHeight * scale) / 2.0f;
        }
    }

}
