package com.example.model.structure;

public class Plate {

    // 顶点
    public static class Vertexes {
        public int[] lt;    // 左上
        public int[] rt;    // 右上
        public int[] rb;    // 右下
        public int[] lb;    // 左下
        public Vertexes() {}
        public Vertexes(int[] lt, int[] rt, int[] rb, int[] lb) {
            this.lt = lt;
            this.rt = rt;
            this.rb = rb;
            this.lb = lb;
        }
    }

}
