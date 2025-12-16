package com.example.model.util;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import com.example.model.FacePlateDetector;
import java.util.List;

public class ModelImageHelper {

    // 根据给定框裁减图像
    public static Bitmap cutBitmapByBox(Bitmap originalBitmap, int[] bbox, int expandPixels) {
        // 各方向扩大指定像素
        int left = Math.max(0, bbox[0] - expandPixels);
        int top = Math.max(0, bbox[1] - expandPixels);
        int right = Math.min(originalBitmap.getWidth(), bbox[2] + expandPixels);
        int bottom = Math.min(originalBitmap.getHeight(), bbox[3] + expandPixels);
        if (right <= left || bottom <= top) {
            return originalBitmap;
        }
        return Bitmap.createBitmap(originalBitmap, left, top, right - left, bottom - top);
    }

    // 归一化
    public static float[] normalizeBitmap(Bitmap bitmap, int modelWidth, int modelHeight, float meanValue, float stdValue) {
        int size = modelWidth * modelHeight;
        int[] pixels = new int[size];
        bitmap.getPixels(pixels, 0, modelWidth, 0, 0, modelWidth, modelHeight);
        float[] inputData = new float[3 * size];
        for (int i = 0; i < pixels.length; i++) {
            int pixel = pixels[i];
            inputData[i] = (((pixel >> 16) & 0xFF) / 255.0f - meanValue) / stdValue;
            inputData[i + size] = (((pixel >> 8) & 0xFF) / 255.0f - meanValue) / stdValue;
            inputData[i + 2 * size] = ((pixel & 0xFF) / 255.0f - meanValue) / stdValue;
        }
        return inputData;
    }

    // 将检测结果绘制到图像上
    public static Bitmap drawDetectionsToBitmap(Bitmap originalBitmap, List<FacePlateDetector.Result> results) {
        Bitmap resultBitmap = originalBitmap.copy(Bitmap.Config.RGB_565, true);
        Canvas canvas = new Canvas(resultBitmap);
        // 创建边界框
        Paint boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(3f);
        boxPaint.setColor(Color.GREEN);
        // 创建特征点
        Paint landmarkPaint = new Paint();
        landmarkPaint.setColor(Color.RED);
        landmarkPaint.setStrokeWidth(5f);
        // 创建文本
        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(24f);
        textPaint.setFakeBoldText(true);
        // 创建背景
        Paint bgPaint = new Paint();
        bgPaint.setColor(Color.BLACK);
        bgPaint.setAlpha(128);
        // 遍历所有检测结果并绘制
        for (FacePlateDetector.Result result : results) {
            int[] bbox = result.bbox;
            float threshold = result.threshold;
            List<float[]> landmarks = result.landmarks;
            RectF rect = new RectF(bbox[0], bbox[1], bbox[2], bbox[3]);
            canvas.drawRect(rect, boxPaint);
            for (float[] landmark : landmarks) {
                canvas.drawPoint(landmark[0], landmark[1], landmarkPaint);
            }
            String label = String.format("threshold: %.2f", threshold);
            float textWidth = textPaint.measureText(label);
            float textHeight = textPaint.getTextSize();
            RectF textBg = new RectF(
                    rect.left,
                    rect.top - textHeight - 10,
                    rect.left + textWidth + 10,
                    rect.top
            );
            canvas.drawRect(textBg, bgPaint);
            canvas.drawText(label, rect.left + 5, rect.top - 5, textPaint);
        }
        return resultBitmap;
    }

}
