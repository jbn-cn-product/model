package com.example.model.onnx.utils;

import com.example.model.onnx.structure.Position.Box;
import com.example.model.onnx.structure.Position.Point;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class DataHelper {

    private static final String TAG = "MyLogcat-DataHelper";

    // 非极大值抑制
    public static void nms(List<?> sortedList, List<Box> boxList, float iouThreshold) {
        Set<Integer> nmsIndexSet = new HashSet<>();
        for (int i = 0; i < sortedList.size() - 1; i++) {
            Box box1 = boxList.get(i);
            for (int j = i + 1; j < sortedList.size(); j++) {
                Box box2 = boxList.get(j);
                if (DataHelper.iou(box1, box2) >= iouThreshold) {
                    nmsIndexSet.add(j);
                }
            }
        }
        List<Integer> nmsIndexList = new ArrayList<>(nmsIndexSet);
        nmsIndexList.sort((a, b) -> Integer.compare(b, a)); // 降序排列
        for (int index : nmsIndexList) {
            sortedList.remove(index);
        }
    }

    // 计算交并比
    public static float iou(Box box1, Box box2) {
        float width = Math.max(0, box1.point.x < box2.point.x ? (box1.point.x + box1.width - box2.point.x) : (box2.point.x + box2.width - box1.point.x));
        float height = Math.max(0, box1.point.y < box2.point.y ? (box1.point.y + box1.height - box2.point.y) : (box2.point.y + box2.height - box1.point.y));
        float area1 = box1.width * box1.height;
        float area2 = box2.width * box2.height;
        float area = width * height;
        float union = area1 + area2 - area;
        if (union <= 0) {
            return 0.0f;
        }
        return area / union;
    }

    // 校正人脸关键点布局
    public static byte[] rectifyFace(byte[] rgbData, int width, int height, List<float[]> landmarks) {
        final double[][] refPoints = {
                {30.29459953f + 8.0f, 51.69630051f}, // 左眼
                {65.53179932f + 8.0f, 51.50139999f}, // 右眼
                {48.02519989f + 8.0f, 71.73660278f}, // 鼻子
                {33.54930115f + 8.0f, 92.36550140f}, // 左嘴角
                {62.72990036f + 8.0f, 92.20410156f}  // 右嘴角
        };
        double[][] aData = new double[10][6];
        double[] bData = new double[10];
        for (int i = 0; i < 5; i++) {
            double x = landmarks.get(i)[0];
            double y = landmarks.get(i)[1];
            int row1 = i * 2;
            int row2 = i * 2 + 1;
            aData[row1][0] = x;
            aData[row1][1] = y;
            aData[row1][2] = 1.0;
            aData[row1][3] = 0.0;
            aData[row1][4] = 0.0;
            aData[row1][5] = 0.0;
            bData[row1] = refPoints[i][0];
            aData[row2][0] = 0.0;
            aData[row2][1] = 0.0;
            aData[row2][2] = 0.0;
            aData[row2][3] = x;
            aData[row2][4] = y;
            aData[row2][5] = 1.0;
            bData[row2] = refPoints[i][1];
        }
        double[] params = new SingularValueDecomposition(new Array2DRowRealMatrix(aData, false)).getSolver().solve(new Array2DRowRealMatrix(bData)).getColumn(0);
        float a = (float) params[0];
        float b = (float) params[1];
        float c = (float) params[2];
        float d = (float) params[3];
        float e = (float) params[4];
        float f = (float) params[5];
        float det = a * e - b * d;
        float invA = e / det;
        float invB = -b / det;
        float invC = (b * f - c * e) / det;
        float invD = -d / det;
        float invE = a / det;
        float invF = (c * d - a * f) / det;
        byte[] resultData = new byte[width * height * 3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float srcX = invA * (float) x + invB * (float) y + invC;
                float srcY = invD * (float) x + invE * (float) y + invF;
                if (srcX < 0.0f || srcX >= width - 1.0f || srcY < 0.0f || srcY >= height - 1.0f) {
                    continue;
                }
                int srcIdx = ((int) srcY * width + (int) srcX) * 3;
                int destIdx = y * width * 3 + x * 3;
                resultData[destIdx]     = rgbData[srcIdx];
                resultData[destIdx + 1] = rgbData[srcIdx + 1];
                resultData[destIdx + 2] = rgbData[srcIdx + 2];
            }
        }
        return resultData;
    }

    // 计算模糊度
    public static double calculateBlur(byte[] rgbData, int width, int height) {
        final double weightR = 0.299;
        final double weightG = 0.587;
        final double weightB = 0.114;
        double sum = 0.0;
        double sumSq = 0.0;
        int validPixelCount = 0;
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int index = (y * width + x) * 3;
                int r = rgbData[index] & 0xFF;
                int g = rgbData[index + 1] & 0xFF;
                int b = rgbData[index + 2] & 0xFF;
                double gray = weightR * r + weightG * g + 0.114 * b;
                double[] neighbors = new double[8];
                // 左上 (y-1, x-1)
                int idxTL = ((y - 1) * width + (x - 1)) * 3;
                neighbors[0] = weightR * (rgbData[idxTL] & 0xFF) + weightG * (rgbData[idxTL + 1] & 0xFF) + weightB * (rgbData[idxTL + 2] & 0xFF);
                // 上 (y-1, x)
                int idxT = ((y - 1) * width + x) * 3;
                neighbors[1] = weightR * (rgbData[idxT] & 0xFF) + weightG * (rgbData[idxT + 1] & 0xFF) + weightB * (rgbData[idxT + 2] & 0xFF);
                // 右上 (y-1, x+1)
                int idxTR = ((y - 1) * width + (x + 1)) * 3;
                neighbors[2] = weightR * (rgbData[idxTR] & 0xFF) + weightG * (rgbData[idxTR + 1] & 0xFF) + weightB * (rgbData[idxTR + 2] & 0xFF);
                // 左 (y, x-1)
                int idxL = (y * width + (x - 1)) * 3;
                neighbors[3] = weightR * (rgbData[idxL] & 0xFF) + weightG * (rgbData[idxL + 1] & 0xFF) + weightB * (rgbData[idxL + 2] & 0xFF);
                // 右 (y, x+1)
                int idxR = (y * width + (x + 1)) * 3;
                neighbors[4] = weightR * (rgbData[idxR] & 0xFF) + weightG * (rgbData[idxR + 1] & 0xFF) + weightB * (rgbData[idxR + 2] & 0xFF);
                // 左下 (y+1, x-1)
                int idxBL = ((y + 1) * width + (x - 1)) * 3;
                neighbors[5] = weightR * (rgbData[idxBL] & 0xFF) + weightG * (rgbData[idxBL + 1] & 0xFF) + weightB * (rgbData[idxBL + 2] & 0xFF);
                // 下 (y+1, x)
                int idxB = ((y + 1) * width + x) * 3;
                neighbors[6] = weightR * (rgbData[idxB] & 0xFF) + weightG * (rgbData[idxB + 1] & 0xFF) + weightB * (rgbData[idxB + 2] & 0xFF);
                // 右下 (y+1, x+1)
                int idxBR = ((y + 1) * width + (x + 1)) * 3;
                neighbors[7] = weightR * (rgbData[idxBR] & 0xFF) + weightG * (rgbData[idxBR + 1] & 0xFF) + weightB * (rgbData[idxBR + 2] & 0xFF);
                double neighborSum = 0;
                for (double n : neighbors) {
                    neighborSum += n;
                }
                double laplacian = 8 * gray - neighborSum;
                sum += laplacian;
                sumSq += laplacian * laplacian;
                validPixelCount++;
            }
        }
        double mean = sum / validPixelCount;
        return (sumSq / validPixelCount) - (mean * mean);
    }

    // 计算坐标在旋转图像后的新值
    public static void rotatePoint(Point point, float degrees, float srcWidth, float srcHeight) {
        if (degrees == 0) {
            return;
        }
        double radians = Math.toRadians(degrees);
        double sin = Math.sin(radians);
        double cos = Math.cos(radians);
        double x0 = 0;
        double y0 = 0;
        double x1 = srcWidth * cos;
        double y1 = -srcWidth * sin;
        double x2 = srcHeight * sin;
        double y2 = srcHeight * cos;
        double x3 = srcWidth * cos + srcHeight * sin;
        double y3 = -srcWidth * sin + srcHeight * cos;
        double minX = Math.min(Math.min(Math.min(x0, x1), x2), x3);
        double minY = Math.min(Math.min(Math.min(y0, y1), y2), y3);
        float rawRotatedX = (float) (point.x * cos + point.y * sin);
        float rawRotatedY = (float) (-point.x * sin + point.y * cos);
        point.x = (float) (rawRotatedX - minX);
        point.y = (float) (rawRotatedY - minY);
    }

}
