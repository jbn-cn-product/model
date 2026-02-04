package com.example.model.utils.android;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import com.example.model.structure.Common.Box;
import com.example.model.structure.Face;
import java.util.Objects;

public class ImageProcesser {

    private static final String TAG = "MyLogcat-ImageProcesser";

    // 缩放
    public static Bitmap resizeBitmap(Bitmap srcBitmap, int targetWidth, int targetHeight, boolean keepScale) {
        int srcWidth = srcBitmap.getWidth();
        int srcHeight = srcBitmap.getHeight();
        // 不需要保持比例，或者两者比例本就相等，直接复制原图返回
        if (!keepScale || Math.abs(((float) srcWidth / srcHeight) - ((float) targetWidth / targetHeight)) < 1e-5f) {
            return Bitmap.createScaledBitmap(srcBitmap, targetWidth, targetHeight, true);
        }
        float scale = Math.min((float) targetWidth / srcWidth, (float) targetHeight / srcHeight); // 获取相对更小的那一个缩放倍率
        Matrix matrix = new Matrix();
        matrix.postScale(scale, scale);
        matrix.postTranslate((targetWidth - srcWidth * scale) / 2.0f, (targetHeight - srcHeight * scale) / 2.0f);
        Bitmap targetBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(targetBitmap);
        canvas.drawColor(Color.rgb(114, 114, 114)); // 填充为灰色，模型推理的标准做法
        canvas.drawBitmap(srcBitmap, matrix, null);
        return targetBitmap;
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

    // RGB转Bitmap
    public static Bitmap convertRGBToBitmap(byte[] rgbData, int width, int height) {
        int[] pixels = new int[width * height];
        int index = 0;
        for (int i = 0; i < pixels.length; i++) {
            int r = rgbData[index++] & 0xFF;
            int g = rgbData[index++] & 0xFF;
            int b = rgbData[index++] & 0xFF;
            pixels[i] = Color.rgb(r, g, b);
        }
        return Bitmap.createBitmap(pixels, 0, width, width, height, Bitmap.Config.ARGB_8888);
    }

    // 旋转图像
    public static Bitmap rotateBitmap(Bitmap source, int angle) {
        if (angle == 0) {
            return source.copy(Objects.requireNonNull(source.getConfig()), true);
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    // 释放图像资源
    public static void recycleBitmap(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    // 用于绘制检测结果
    public static class DrawPaints {
        private static Canvas canvas;       // 画布
        private static Paint boxPaint;      // 边界框
        private static Paint textBgPaint;   // 文本背景
        private static Paint textPaint;     // 文本
        private static Paint pointPaint;    // 关键点
        public static Bitmap createDrawBitmap(Bitmap bitmap) {
            Bitmap drawBitmap = bitmap.copy(Bitmap.Config.RGB_565, true);
            canvas = new Canvas(drawBitmap);
            boxPaint = new Paint();
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(3f);
            boxPaint.setColor(Color.GREEN);
            textBgPaint = new Paint();
            textBgPaint.setColor(Color.BLACK);
            textBgPaint.setAlpha(128);
            textPaint = new Paint();
            textPaint.setColor(Color.WHITE);
            textPaint.setTextSize(24f);
            textPaint.setFakeBoldText(true);
            pointPaint = new Paint();
            pointPaint.setColor(Color.RED);
            pointPaint.setStrokeWidth(5f);
            return drawBitmap;
        }
        public static void drawDetection(Box box, float confidence, Face.Landmarks landmarks) {
            RectF boxRectF = new RectF(box.point.x, box.point.y, box.point.x + box.width, box.point.y + box.height);
            canvas.drawRect(boxRectF, boxPaint);
            String label = String.format("%.2f", confidence);
            canvas.drawRect(new RectF(
                    boxRectF.left,
                    boxRectF.top - textPaint.getTextSize() - 10,
                    boxRectF.left + textPaint.measureText(label) + 10,
                    boxRectF.top
            ), textBgPaint);
            canvas.drawText(label, boxRectF.left + 5, boxRectF.top - 5, textPaint);
            if (landmarks != null) {
                canvas.drawPoint(landmarks.leftEye.x, landmarks.leftEye.y, pointPaint);
                canvas.drawPoint(landmarks.rightEye.x, landmarks.rightEye.y, pointPaint);
                canvas.drawPoint(landmarks.nose.x, landmarks.nose.y, pointPaint);
                canvas.drawPoint(landmarks.leftMouth.x, landmarks.leftMouth.y, pointPaint);
                canvas.drawPoint(landmarks.rightMouth.x, landmarks.rightMouth.y, pointPaint);
            }
        }
    }

}
