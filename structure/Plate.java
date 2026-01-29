package com.example.model.structure;

import com.example.model.structure.Common.Point;

public class Plate {

    // 顶点
    public static class Vertexes {
        public Point lt;    // 左上
        public Point rt;    // 右上
        public Point rb;    // 右下
        public Point lb;    // 左下
        public Vertexes() {}
        public Vertexes(Point lt, Point rt, Point rb, Point lb) {
            this.lt = lt;
            this.rt = rt;
            this.rb = rb;
            this.lb = lb;
        }
    }

}
