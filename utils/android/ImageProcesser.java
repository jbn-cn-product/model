package com.example.model.utils.android;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import com.example.model.structure.Common.Box;

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
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    // 裁切图像
    public static Bitmap cutBitmap(Bitmap bitmap, Box box) {
        int bitmapWidth = bitmap.getWidth();
        int bitmapHeight = bitmap.getHeight();
        int x = (int) box.point.x;
        int y = (int) box.point.y;
        int w = (int) box.width;
        if (x + w > bitmapWidth) {
            w = bitmapWidth - x;
        }
        int h = (int) box.height;
        if (y + h > bitmapHeight) {
            h = bitmapHeight - y;
        }
        return Bitmap.createBitmap(bitmap, x, y, w, h);
    }

    // 释放图像资源
    public static void recycleBitmap(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

}
