package com.example.model.core;

import com.example.model.core.base.OnnxDeployer;
import com.example.model.structure.Common;
import com.example.model.structure.Face;
import com.example.model.structure.Plate;
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

    public static class Result {
        public Common.Box box;                  // 检测框
        public float confidence;                // 置信度
        public int classId;                     // 类别 0-车牌 1-人脸
        public Face.Angles faceAngles;          // 人脸姿态
        public Face.Landmarks faceLandmarks;    // 人脸关键点
        public Plate.Vertexes plateVertexes;    // 车牌顶点
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
        List<float[]> boxList = new ArrayList<>();
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
            float[] box = new float[4];
            box[0] = x - w / 2;
            box[1] = y - h / 2;
            box[2] = x + w / 2;
            box[3] = y + h / 2;
            boxList.add(box);
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
                float[] box1 = boxList.get(current);
                float[] box2 = boxList.get(it.next());
                float x1 = Math.max(box1[0], box2[0]);
                float x2 = Math.min(box1[2], box2[2]);
                float y1 = Math.max(box1[1], box2[1]);
                float y2 = Math.min(box1[3], box2[3]);
                float interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
                if (interArea > 0) {
                    float iou = interArea / ((box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - interArea);
                    if (iou > IOU_THRESHOLD) {
                        it.remove();
                    }
                }
            }
        }
        // 创建结果
        List<Result> results = new ArrayList<>();
        for (int keepIndex : keepIndexes) {
            float[] box = boxList.get(keepIndex);
            Result result = new Result();
            result.box = new Common.Box((int) box[0], (int) box[1], (int) box[2], (int) box[3]);
            result.confidence = confidenceList.get(keepIndex);
            result.classId = classIdList.get(keepIndex);
            float[] point = pointList.get(keepIndex);
            List<int[]> points = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                int x = (int) point[i * 2];
                int y = (int) point[i * 2 + 1];
                points.add(new int[]{x, y});
            }
            if (result.classId == 0) {
                result.plateVertexes = new Plate.Vertexes();
                // 判断位置
                List<int[]> vertexes = new ArrayList<>(points);
                vertexes.remove(2); // 第3个关键点在车牌中无意义
                vertexes.sort((p1, p2) -> Float.compare(p1[0], p2[0]));
                result.plateVertexes.lt = vertexes.get(0)[1] < vertexes.get(1)[1] ? vertexes.get(0) : vertexes.get(1);
                result.plateVertexes.rt = vertexes.get(2)[1] < vertexes.get(3)[1] ? vertexes.get(2) : vertexes.get(3);
                result.plateVertexes.rb = vertexes.get(2)[1] >= vertexes.get(3)[1] ? vertexes.get(2) : vertexes.get(3);
                result.plateVertexes.lb = vertexes.get(0)[1] >= vertexes.get(1)[1] ? vertexes.get(0) : vertexes.get(1);
            } else if (result.classId == 1) {
                result.faceLandmarks = new Face.Landmarks();
                // 存入坐标点
                result.faceLandmarks.leftEye = points.get(0);
                result.faceLandmarks.rightEye = points.get(1);
                result.faceLandmarks.nose = points.get(2);
                result.faceLandmarks.leftMouth = points.get(3);
                result.faceLandmarks.rightMouth = points.get(4);
                // 计算角度
                result.faceAngles = new Face.Angles();
                result.faceAngles.yaw = 0.0f;
                result.faceAngles.pitch = 0.0f;
                result.faceAngles.roll = 0.0f;
                float eyeCenterX = (result.faceLandmarks.leftEye[0] + result.faceLandmarks.rightEye[0]) / 2.0f;
                float eyeDistance = Math.abs(result.faceLandmarks.rightEye[0] - result.faceLandmarks.leftEye[0]);
                float noseOffset = result.faceLandmarks.nose[0] - eyeCenterX;
                if (eyeDistance > 0.0f) {
                    result.faceAngles.yaw = Math.max(-1.0f, Math.min(1.0f, noseOffset / eyeDistance)) * 30.0f;
                }
                float eyeCenterY = (result.faceLandmarks.leftEye[1] + result.faceLandmarks.rightEye[1]) / 2.0f;
                float mouthCenterY = (result.faceLandmarks.leftMouth[1] + result.faceLandmarks.rightMouth[1]) / 2.0f;
                float noseDeviation = result.faceLandmarks.nose[1] - ((eyeCenterY + mouthCenterY) / 2.0f);
                float faceHeight = Math.abs(mouthCenterY - eyeCenterY);
                if (faceHeight > 0.0f) {
                    result.faceAngles.pitch = Math.max(-1.0f, Math.min(1.0f, noseDeviation / faceHeight)) * 20.0f;
                }
                float eyeDx = result.faceLandmarks.rightEye[0] - result.faceLandmarks.leftEye[0];
                float eyeDy = result.faceLandmarks.rightEye[1] - result.faceLandmarks.leftEye[1];
                if (Math.abs(eyeDx) > 0.0f) {
                    result.faceAngles.roll = (float) Math.toDegrees(Math.atan(eyeDy / eyeDx));
                }
            }
            results.add(result);
        }
        return results;
    }

}
