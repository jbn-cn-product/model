package com.example.model.utils;

import com.example.model.core.detector.FacePlateDetector;
import com.example.model.structure.Common.Box;
import com.example.model.structure.Face;
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

    // 计算姿态
    public static Face.Angles calculateAngles(Face.Landmarks landmarks) {
        Face.Angles angles = new Face.Angles(0.0f, 0.0f, 0.0f);
        float eyeCenterX = (landmarks.leftEye.x + landmarks.rightEye.x) / 2.0f;
        float eyeDistance = Math.abs(landmarks.rightEye.x - landmarks.leftEye.x);
        float noseOffset = landmarks.nose.x - eyeCenterX;
        if (eyeDistance > 0.0f) {
            angles.yaw = Math.max(-1.0f, Math.min(1.0f, noseOffset / eyeDistance)) * 30.0f;
        }
        float eyeCenterY = (landmarks.leftEye.y + landmarks.rightEye.y) / 2.0f;
        float mouthCenterY = (landmarks.leftMouth.y + landmarks.rightMouth.y) / 2.0f;
        float noseDeviation = landmarks.nose.y - ((eyeCenterY + mouthCenterY) / 2.0f);
        float faceHeight = Math.abs(mouthCenterY - eyeCenterY);
        if (faceHeight > 0.0f) {
            angles.pitch = Math.max(-1.0f, Math.min(1.0f, noseDeviation / faceHeight)) * 20.0f;
        }
        float eyeDx = landmarks.rightEye.x - landmarks.leftEye.x;
        float eyeDy = landmarks.rightEye.y - landmarks.leftEye.y;
        if (Math.abs(eyeDx) > 0.0f) {
            angles.roll = (float) Math.toDegrees(Math.atan(eyeDy / eyeDx));
        }
        return angles;
    }

    // 根据关键点校正人脸
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

    // 计算人脸模糊度
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

    // 将模型输出坐标还原到原图尺寸上
    public static void restoreResultCoordinates(List<FacePlateDetector.Result> results, int imageWidth, int imageHeight, int modelWidth, int modelHeight) {
        float scale = Math.min((float) modelWidth / imageWidth, (float) modelHeight / imageHeight);
        float offsetX = (modelWidth - imageWidth * scale) / 2.0f;
        float offsetY = (modelHeight - imageHeight * scale) / 2.0f;
        for (FacePlateDetector.Result result : results) {
            result.box.point.x = (result.box.point.x - offsetX) / scale;
            result.box.point.y = (result.box.point.y - offsetY) / scale;
            result.box.width = result.box.width / scale;
            result.box.height = result.box.height / scale;
            if (result.classId == 0) {
                result.plateVertexes.lt.x = (result.plateVertexes.lt.x - offsetX) / scale;
                result.plateVertexes.lt.y = (result.plateVertexes.lt.y - offsetY) / scale;
                result.plateVertexes.rt.x = (result.plateVertexes.rt.x - offsetX) / scale;
                result.plateVertexes.rt.y = (result.plateVertexes.rt.y - offsetY) / scale;
                result.plateVertexes.rb.x = (result.plateVertexes.rb.x - offsetX) / scale;
                result.plateVertexes.rb.y = (result.plateVertexes.rb.y - offsetY) / scale;
                result.plateVertexes.lb.x = (result.plateVertexes.lb.x - offsetX) / scale;
                result.plateVertexes.lb.y = (result.plateVertexes.lb.y - offsetY) / scale;
            } else if (result.classId == 1) {
                result.faceLandmarks.leftEye.x = (result.faceLandmarks.leftEye.x - offsetX) / scale;
                result.faceLandmarks.leftEye.y = (result.faceLandmarks.leftEye.y - offsetY) / scale;
                result.faceLandmarks.rightEye.x = (result.faceLandmarks.rightEye.x - offsetX) / scale;
                result.faceLandmarks.rightEye.y = (result.faceLandmarks.rightEye.y - offsetY) / scale;
                result.faceLandmarks.nose.x = (result.faceLandmarks.nose.x - offsetX) / scale;
                result.faceLandmarks.nose.y = (result.faceLandmarks.nose.y - offsetY) / scale;
                result.faceLandmarks.leftMouth.x = (result.faceLandmarks.leftMouth.x - offsetX) / scale;
                result.faceLandmarks.leftMouth.y = (result.faceLandmarks.leftMouth.y - offsetY) / scale;
                result.faceLandmarks.rightMouth.x = (result.faceLandmarks.rightMouth.x - offsetX) / scale;
                result.faceLandmarks.rightMouth.y = (result.faceLandmarks.rightMouth.y - offsetY) / scale;
            }
        }
    }

}
