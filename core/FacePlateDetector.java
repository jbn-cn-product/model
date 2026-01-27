package com.example.model.core;

import com.example.model.core.base.OnnxDeployer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public class FacePlateDetector extends OnnxDeployer<List<FacePlateDetector.Result>> {

    public static class Box {
        public int x;
        public int y;
        public int w;
        public int h;
    }
    public static class Face {
        public float[] landmarkLeftEye;     // 左眼
        public float[] landmarkRightEye;    // 右眼
        public float[] landmarkNose;        // 鼻子
        public float[] landmarkLeftMouth;   // 左嘴角
        public float[] landmarkRightMouth;  // 右嘴角
        public float yaw;                   // 偏航角
        public float pitch;                 // 俯仰角
        public float roll;                  // 翻滚角
    }
    public static class Plate {
        public float[] vertexLeftTop;       // 左上顶点
        public float[] vertexRightTop;      // 右上顶点
        public float[] vertexRightBottom;   // 右下顶点
        public float[] vertexLeftBottom;    // 左下顶点
    }

    public static class Result {
        public Box box;             // 检测框
        public float confidence;    // 置信度
        public int classId;         // 类别 0-车牌 1-人脸
        public Face face;           // 人脸特征
        public Plate plate;         // 车牌特征
    }

    private static final String TAG = "MyLogcat-FacePlateDetector";

    // 模型参数
    public static final int MODEL_WIDTH = 640;
    public static final int MODEL_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.3f;   // 置信度阈值
    private static final float IOU_THRESHOLD = 0.5f;    // 重合阈值
    private static final float MEAN_VALUE = 0.0f;
    private static final float STD_VALUE = 1.5f;

    public FacePlateDetector(Logger logger, byte[] modelData) {
        super(logger, modelData, MODEL_WIDTH, MODEL_HEIGHT, MEAN_VALUE, STD_VALUE);
    }

    // 运行
    public List<Result> run(byte[] rgbData) {
        long startTime = System.currentTimeMillis();
        List<Result> results = super.inference(rgbData);
        if (!results.isEmpty()) {
            logger.debug(TAG, String.format("run detection in %d ms, %d objects", System.currentTimeMillis() - startTime, results.size()));
        }
        return results;
    }

    // 后处理
    @Override
    protected List<Result> postprocess(OrtSession.Result sessionResult) {
        // 处理输出张量
        long[] outputShape;
        FloatBuffer buffer;
        try (OnnxTensor tensor = (OnnxTensor) sessionResult.get(0)) {
            TensorInfo tensorInfo = (TensorInfo) super.session.getOutputInfo().values().iterator().next().getInfo();
            buffer = tensor.getFloatBuffer();
            outputShape = tensorInfo.getShape();
        } catch (OrtException e) {
            logger.error(TAG, "update output tensor failed: " + e.getMessage());
            return Collections.emptyList();
        }
        float[] outputArray = new float[buffer.remaining()];
        buffer.get(outputArray);
        int featuresPerDetection = (int) outputShape[2];
        int totalDetections = (int) outputShape[0] * (int) outputShape[1];
        float[][] outputs = new float[totalDetections][featuresPerDetection];
        for (int i = 0; i < totalDetections; i++) {
            System.arraycopy(outputArray, i * featuresPerDetection, outputs[i], 0, featuresPerDetection);
        }
        return decodeResult(outputs);
    }

    // 解码检测结果
    private List<Result> decodeResult(float[][] outputs) {
        List<float[]> bboxList = new ArrayList<>();
        List<Float> confidenceList = new ArrayList<>();
        List<Integer> classIdList = new ArrayList<>();
        List<float[]> pointList = new ArrayList<>();
        // 解析输出结构
        for (float[] output : outputs) {
            float confidence = output[4];
            float plateScore = output[15];
            float faceScore = output[16];
            float finalConfidence = confidence * Math.max(plateScore, faceScore); // 输出置信度*最高类别分数=最终置信度
            if (finalConfidence < CONF_THRESHOLD) {
                continue;
            }
            confidenceList.add(finalConfidence);
            classIdList.add(plateScore > faceScore ? 0 : 1);
            float x = output[0], y = output[1], w = output[2], h = output[3];
            float[] bbox = new float[4];
            bbox[0] = x - w / 2;
            bbox[1] = y - h / 2;
            bbox[2] = x + w / 2;
            bbox[3] = y + h / 2;
            bboxList.add(bbox);
            float[] point = new float[10];
            for (int i = 0; i < 10; i += 2) {
                point[i] = output[5 + i];
                point[i + 1] = output[6 + i];
            }
            pointList.add(point);
        }
        // NMS过滤多个重复检测框
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < confidenceList.size(); i++) {
            order.add(i);
        }
        order.sort((a, b) -> Float.compare(confidenceList.get(b), confidenceList.get(a)));
        List<Integer> keepIndexes = new ArrayList<>();
        while (!order.isEmpty()) {
            int current = order.remove(0);
            keepIndexes.add(current);
            Iterator<Integer> it = order.iterator();
            while (it.hasNext()) {
                float[] bbox1 = bboxList.get(current);
                float[] bbox2 = bboxList.get(it.next());
                float x1 = Math.max(bbox1[0], bbox2[0]);
                float x2 = Math.min(bbox1[2], bbox2[2]);
                float y1 = Math.max(bbox1[1], bbox2[1]);
                float y2 = Math.min(bbox1[3], bbox2[3]);
                float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
                if (interArea > 0) {
                    float iou = interArea / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - interArea);
                    if (iou > IOU_THRESHOLD) {
                        it.remove();
                    }
                }
            }
        }
        // 创建结果
        List<Result> results = new ArrayList<>();
        for (int keepIndex : keepIndexes) {
            float[] bbox = bboxList.get(keepIndex);
            Result result = new Result();
            result.box = new Box();
            result.box.x = (int) bbox[0];
            result.box.y = (int) bbox[1];
            result.box.w = (int) bbox[2];
            result.box.h = (int) bbox[3];
            result.confidence = confidenceList.get(keepIndex);
            result.classId = classIdList.get(keepIndex);
            float[] point = pointList.get(keepIndex);
            List<float[]> points = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                points.add(new float[]{point[i * 2], point[i * 2 + 1]});
            }
            if (result.classId == 0) {
                result.plate = new Plate();
                // 判断位置
                List<float[]> vertexes = new ArrayList<>(points);
                vertexes.remove(2); // 第3个关键点在车牌中无意义
                vertexes.sort((p1, p2) -> Float.compare(p1[0], p2[0]));
                result.plate.vertexLeftTop = vertexes.get(0)[1] < vertexes.get(1)[1] ? vertexes.get(0) : vertexes.get(1);
                result.plate.vertexRightTop = vertexes.get(2)[1] < vertexes.get(3)[1] ? vertexes.get(2) : vertexes.get(3);
                result.plate.vertexRightBottom = vertexes.get(2)[1] >= vertexes.get(3)[1] ? vertexes.get(2) : vertexes.get(3);
                result.plate.vertexLeftBottom = vertexes.get(0)[1] >= vertexes.get(1)[1] ? vertexes.get(0) : vertexes.get(1);
            } else if (result.classId == 1) {
                result.face = new Face();
                // 存入坐标点
                result.face.landmarkLeftEye = points.get(0);
                result.face.landmarkRightEye = points.get(1);
                result.face.landmarkNose = points.get(2);
                result.face.landmarkLeftMouth = points.get(3);
                result.face.landmarkRightMouth = points.get(4);
                // 计算角度
                result.face.yaw = 0.0f;
                result.face.pitch = 0.0f;
                result.face.roll = 0.0f;
                float eyeCenterX = (result.face.landmarkLeftEye[0] + result.face.landmarkRightEye[0]) / 2.0f;
                float eyeDistance = Math.abs(result.face.landmarkRightEye[0] - result.face.landmarkLeftEye[0]);
                float noseOffset = result.face.landmarkNose[0] - eyeCenterX;
                if (eyeDistance > 0.0f) {
                    result.face.yaw = Math.max(-1.0f, Math.min(1.0f, noseOffset / eyeDistance)) * 30.0f;
                }
                float eyeCenterY = (result.face.landmarkLeftEye[1] + result.face.landmarkRightEye[1]) / 2.0f;
                float mouthCenterY = (result.face.landmarkLeftMouth[1] + result.face.landmarkRightMouth[1]) / 2.0f;
                float noseDeviation = result.face.landmarkNose[1] - ((eyeCenterY + mouthCenterY) / 2.0f);
                float faceHeight = Math.abs(mouthCenterY - eyeCenterY);
                if (faceHeight > 0.0f) {
                    result.face.pitch = Math.max(-1.0f, Math.min(1.0f, noseDeviation / faceHeight)) * 20.0f;
                }
                float eyeDx = result.face.landmarkRightEye[0] - result.face.landmarkLeftEye[0];
                float eyeDy = result.face.landmarkRightEye[1] - result.face.landmarkLeftEye[1];
                if (Math.abs(eyeDx) > 0.0f) {
                    result.face.roll = (float) Math.toDegrees(Math.atan(eyeDy / eyeDx));
                }
            }
            results.add(result);
        }
        return results;
    }

}
