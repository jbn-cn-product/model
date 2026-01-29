package com.example.model.utils.android;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import com.example.model.core.FacePlateDetector;
import com.example.model.structure.Common;

import java.util.List;

public class ImageProcesser {

    private static final String TAG = "MyLogcat-ImageProcesser";

    // 缩放
    public static Bitmap resizeBitmap(Bitmap bitmap, int width, int height, boolean keepScale) {
        if (keepScale) {
            // Bitmap自带的缩放方法处理的图像会变形，导致关键点错位，在右边或下方填充黑色区域以保持原始比例
            Bitmap resizedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
            float scale = Math.min((float) width / bitmap.getWidth(), (float) height / bitmap.getHeight());
            Canvas canvas = new Canvas(resizedBitmap);
            Matrix matrix = new Matrix();
            matrix.postScale(scale, scale);
            matrix.postTranslate(0, 0);
            canvas.drawBitmap(bitmap, matrix, null);
            return resizedBitmap;
        } else {
            return Bitmap.createScaledBitmap(bitmap, width, height, true);
        }
    }

    // Bitmap转RGB
    public static byte[] convertBitmapToRGB(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int size = width * height;
        int[] pixels = new int[size];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
        byte[] data = new byte[size * 3];
        int index = 0;
        for (int pixel : pixels) {
            data[index++] = (byte) Color.red(pixel);
            data[index++] = (byte) Color.green(pixel);
            data[index++] = (byte) Color.blue(pixel);
        }
        return data;
    }

    // 根据给定框裁减图像
    public static Bitmap cutBitmapByBox(Bitmap bitmap, Common.Box box, int expandPixels) {
        // 各方向扩大指定像素
        int left = Math.max(0, box.x - expandPixels);
        int top = Math.max(0, box.y - expandPixels);
        int right = Math.min(bitmap.getWidth(), box.w + expandPixels);
        int bottom = Math.min(bitmap.getHeight(), box.h + expandPixels);
        if (right <= left || bottom <= top) {
            return bitmap;
        }
        return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top);
    }

    // 将检测结果绘制到图像上
    public static Bitmap drawDetectionsToBitmap(Bitmap bitmap, List<FacePlateDetector.Result> results) {
        Bitmap resultBitmap = bitmap.copy(Bitmap.Config.RGB_565, true);
        Canvas canvas = new Canvas(resultBitmap);
        // 配置边界框
        Paint boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(3f);
        boxPaint.setColor(Color.GREEN);
        // 配置文本
        Paint textBgPaint = new Paint();
        textBgPaint.setColor(Color.BLACK);
        textBgPaint.setAlpha(128);
        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(24f);
        textPaint.setFakeBoldText(true);
        // 配置关键点
        Paint pointPaint = new Paint();
        pointPaint.setColor(Color.RED);
        pointPaint.setStrokeWidth(5f);
        for (FacePlateDetector.Result result : results) {
            // 绘制边界框
            Common.Box box = result.box;
            RectF boxRectF = new RectF(box.x, box.y, box.w, box.h);
            canvas.drawRect(boxRectF, boxPaint);
            // 绘制文本
            String label = String.format("%.2f", result.confidence);
            RectF textBg = new RectF(
                    boxRectF.left,
                    boxRectF.top - textPaint.getTextSize() - 10,
                    boxRectF.left + textPaint.measureText(label) + 10,
                    boxRectF.top
            );
            canvas.drawRect(textBg, textBgPaint);
            canvas.drawText(label, boxRectF.left + 5, boxRectF.top - 5, textPaint);
            // 绘制关键点
            if (result.classId == 0) {
                canvas.drawPoint(result.plateVertexes.lt[0], result.plateVertexes.lt[1], pointPaint);
                canvas.drawPoint(result.plateVertexes.rt[0], result.plateVertexes.rt[1], pointPaint);
                canvas.drawPoint(result.plateVertexes.rb[0], result.plateVertexes.rb[1], pointPaint);
                canvas.drawPoint(result.plateVertexes.lb[0], result.plateVertexes.lb[1], pointPaint);
            } else if (result.classId == 1) {
                canvas.drawPoint(result.faceLandmarks.leftEye[0], result.faceLandmarks.leftEye[1], pointPaint);
                canvas.drawPoint(result.faceLandmarks.rightEye[0], result.faceLandmarks.rightEye[1], pointPaint);
                canvas.drawPoint(result.faceLandmarks.nose[0], result.faceLandmarks.nose[1], pointPaint);
                canvas.drawPoint(result.faceLandmarks.leftMouth[0], result.faceLandmarks.leftMouth[1], pointPaint);
                canvas.drawPoint(result.faceLandmarks.rightMouth[0], result.faceLandmarks.rightMouth[1], pointPaint);
            }
        }
        return resultBitmap;
    }

}
