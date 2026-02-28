package com.example.model.onnx.utils.android;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import com.example.model.onnx.structure.Position.Box;
import com.example.model.onnx.structure.Face;
import com.example.model.onnx.structure.Plate;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

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

    // Bitmap转字节流
    public static byte[] convertBitmapToByteArray(Bitmap bitmap) {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
        return stream.toByteArray();
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
    public static Bitmap rotateBitmap(Bitmap source, float degrees) {
        if (degrees == 0) {
            return source.copy(Objects.requireNonNull(source.getConfig()), true);
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }

    // 释放图像资源
    public static void recycleBitmap(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    // 车牌四点变换
    public static Bitmap rectifyPlate(Bitmap bitmap, Plate plate) {
        Mat srcMat = new Mat();
        Utils.bitmapToMat(bitmap, srcMat);
        Imgproc.cvtColor(srcMat, srcMat, Imgproc.COLOR_RGBA2BGR);
        List<Point> srcPoints = new ArrayList<>();
        srcPoints.add(new Point(plate.lt.x, plate.lt.y));
        srcPoints.add(new Point(plate.rt.x, plate.rt.y));
        srcPoints.add(new Point(plate.rb.x, plate.rb.y));
        srcPoints.add(new Point(plate.lb.x, plate.lb.y));
        MatOfPoint2f srcMatPt = new MatOfPoint2f();
        srcMatPt.fromList(srcPoints);
        double widthTop = Math.sqrt(Math.pow(plate.rt.x - plate.lt.x, 2) + Math.pow(plate.rt.y - plate.lt.y, 2));
        double widthBottom = Math.sqrt(Math.pow(plate.rb.x - plate.lb.x, 2) + Math.pow(plate.rb.y - plate.lb.y, 2));
        int maxWidth = (int) Math.max(widthTop, widthBottom);
        double heightLeft = Math.sqrt(Math.pow(plate.lb.x - plate.lt.x, 2) + Math.pow(plate.lb.y - plate.lt.y, 2));
        double heightRight = Math.sqrt(Math.pow(plate.rb.x - plate.rt.x, 2) + Math.pow(plate.rb.y - plate.rt.y, 2));
        int maxHeight = (int) Math.max(heightLeft, heightRight);
        List<Point> dstPoints = new ArrayList<>();
        dstPoints.add(new Point(0, 0));
        dstPoints.add(new Point(maxWidth, 0));
        dstPoints.add(new Point(maxWidth, maxHeight));
        dstPoints.add(new Point(0, maxHeight));
        MatOfPoint2f dstMatPt = new MatOfPoint2f();
        dstMatPt.fromList(dstPoints);
        Mat perspectiveMatrix = Imgproc.getPerspectiveTransform(srcMatPt, dstMatPt);
        Mat dstMat = new Mat(maxHeight, maxWidth, CvType.CV_8UC3);
        Imgproc.warpPerspective(srcMat, dstMat, perspectiveMatrix, dstMat.size(), Imgproc.INTER_CUBIC);
        Imgproc.cvtColor(dstMat, dstMat, Imgproc.COLOR_BGR2RGBA);
        Bitmap resultBitmap = Bitmap.createBitmap(maxWidth, maxHeight, Bitmap.Config.RGB_565);
        Utils.matToBitmap(dstMat, resultBitmap);
        srcMat.release();
        dstMat.release();
        perspectiveMatrix.release();
        srcMatPt.release();
        dstMatPt.release();
        return resultBitmap;
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
        public static void drawDetection(Box box, float confidence, Face face, Plate plate) {
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
            if (face != null) {
                canvas.drawPoint(face.leftEye.x, face.leftEye.y, pointPaint);
                canvas.drawPoint(face.rightEye.x, face.rightEye.y, pointPaint);
                canvas.drawPoint(face.nose.x, face.nose.y, pointPaint);
                canvas.drawPoint(face.leftMouth.x, face.leftMouth.y, pointPaint);
                canvas.drawPoint(face.rightMouth.x, face.rightMouth.y, pointPaint);
            }
            if (plate != null) {
                canvas.drawPoint(plate.lt.x, plate.lt.y, pointPaint);
                canvas.drawPoint(plate.rt.x, plate.rt.y, pointPaint);
                canvas.drawPoint(plate.rb.x, plate.rb.y, pointPaint);
                canvas.drawPoint(plate.lb.x, plate.lb.y, pointPaint);
            }
        }
    }

}
