package com.example.model.utils.android;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import com.example.model.core.FacePlateDetector;
import java.util.List;

public class ImageProcesser {

    private static final String TAG = "MyLogcat-ImageHelper";

    // 缩放
    public static Bitmap resizeBitmap(Bitmap bitmap, int width, int height, boolean keepScale) {
        if (keepScale) {
            // Bitmap自带的缩放方法处理的图像会变形，导致关键点错位，需要填充以保持原始比例
            Bitmap resizedBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.RGB_565);
            float scale = Math.min((float) width / bitmap.getWidth(), (float) height / bitmap.getHeight());
            int newWidth = Math.round(bitmap.getWidth() * scale);
            int newHeight = Math.round(bitmap.getHeight() * scale);
            Canvas canvas = new Canvas(resizedBitmap);
            Matrix matrix = new Matrix();
            matrix.postScale(scale, scale);
            float left = (width - newWidth) / 2.0f;
            float top = (height - newHeight) / 2.0f;
            matrix.postTranslate(left, top);
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
    public static Bitmap cutBitmapByBox(Bitmap bitmap, int[] bbox, int expandPixels) {
        // 各方向扩大指定像素
        int left = Math.max(0, bbox[0] - expandPixels);
        int top = Math.max(0, bbox[1] - expandPixels);
        int right = Math.min(bitmap.getWidth(), bbox[2] + expandPixels);
        int bottom = Math.min(bitmap.getHeight(), bbox[3] + expandPixels);
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
        Paint bboxPaint = new Paint();
        bboxPaint.setStyle(Paint.Style.STROKE);
        bboxPaint.setStrokeWidth(3f);
        bboxPaint.setColor(Color.GREEN);
        // 配置文本
        Paint textBgPaint = new Paint();
        textBgPaint.setColor(Color.BLACK);
        textBgPaint.setAlpha(128);
        Paint textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(24f);
        textPaint.setFakeBoldText(true);
        // 配置关键点
        Paint landmarksPaint = new Paint();
        landmarksPaint.setColor(Color.RED);
        landmarksPaint.setStrokeWidth(5f);
        for (FacePlateDetector.Result result : results) {
            // 绘制边界框
            int[] bbox = result.bbox;
            RectF bboxRectF = new RectF(bbox[0], bbox[1], bbox[2], bbox[3]);
            canvas.drawRect(bboxRectF, bboxPaint);
            // 绘制文本
            String label = String.format("confidence: %.2f", result.confidence);
            RectF textBg = new RectF(
                    bboxRectF.left,
                    bboxRectF.top - textPaint.getTextSize() - 10,
                    bboxRectF.left + textPaint.measureText(label) + 10,
                    bboxRectF.top
            );
            canvas.drawRect(textBg, textBgPaint);
            canvas.drawText(label, bboxRectF.left + 5, bboxRectF.top - 5, textPaint);
            if (result.classId == 1) {
                // 绘制关键点
                for (float[] landmark : result.landmarks) {
                    canvas.drawPoint(landmark[0], landmark[1], landmarksPaint);
                }
            }
        }
        return resultBitmap;
    }

}
