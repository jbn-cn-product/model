package com.example.model.structure;

import com.example.model.structure.Position.Point;

public class Plate {

    public Point lt;        // 左上
    public Point rt;        // 右上
    public Point rb;        // 右下
    public Point lb;        // 左下

    public double angle;    // 角度

    public Plate(Point lt, Point rt, Point rb, Point lb) {
        this.lt = lt;
        this.rt = rt;
        this.rb = rb;
        this.lb = lb;
        // 计算角度
        float topCenterX = (lt.x + rt.x) / 2.0f;
        float topCenterY = (lt.y + rt.y) / 2.0f;
        float bottomCenterX = (lb.x + rb.x) / 2.0f;
        float bottomCenterY = (lb.y + rb.y) / 2.0f;
        float dy = bottomCenterY - topCenterY;
        float dx = bottomCenterX - topCenterX;
        double angleRad = Math.atan2(dx, dy);
        angle = Math.toDegrees(angleRad);
    }

}
