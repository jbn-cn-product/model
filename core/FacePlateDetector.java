package com.example.model.core;

import com.example.model.core.base.OnnxDeployer;
import com.example.model.structure.Common.Box;
import com.example.model.structure.Common.Point;
import com.example.model.structure.Face;
import com.example.model.structure.Plate;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

public class FacePlateDetector extends OnnxDeployer<List<FacePlateDetector.Result>> {

    public static class Result {
        public Box box;                         // 检测框
        public float confidence;                // 置信度
        public int classId;                     // 类别 0-车牌 1-人脸
        public Face.Angles faceAngles;          // 人脸姿态
        public Face.Landmarks faceLandmarks;    // 人脸关键点
        public Plate.Vertexes plateVertexes;    // 车牌顶点
    }

    private static final String TAG = "MyLogcat-FacePlateDetector";

    // 模型配置
    public static final String MODEL_NAME = "car_face_det.onnx";
    public static final int MODEL_WIDTH = 640;
    public static final int MODEL_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.3f;   // 置信度阈值
    private static final float IOU_THRESHOLD = 0.5f;    // 重合阈值
    private static final float MEAN_VALUE = 0.0f;
    private static final float STD_VALUE = 1.5f;

    public FacePlateDetector(Logger logger, byte[] modelData) {
        super(logger, new Model(modelData, MODEL_WIDTH, MODEL_HEIGHT, MEAN_VALUE, STD_VALUE));
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
        List<Result> results = new ArrayList<>();
        // 解析输出结构
        for (float[] output : outputs) {
            float confidence = output[4];
            if (confidence < CONF_THRESHOLD) { // 先忽略大多数低置信度结果
                continue;
            }
            Result result = new Result();
            // 置信度
            result.confidence = confidence;
            int x = (int) output[0], y = (int) output[1], w = (int) output[2], h = (int) output[3];
            // 检测框
            result.box = new Box(new Point(x - w / 2, y - h / 2), x + w / 2, y + h / 2); // 原始数据的xy坐标是检测框的中心点，需要修改为左上角
            // 类别
            float plateScore = output[15];
            float faceScore = output[16];
            result.classId = plateScore > faceScore ? 0 : 1;
            // 关键点
            List<Point> pointList = new ArrayList<>(5);
            for (int i = 5; i < 14; i += 2) {
                pointList.add(new Point((int) output[i], (int) output[i + 1]));
            }
            if (result.classId == 0) {
                result.plateVertexes = new Plate.Vertexes();
                List<Point> vertexes = new ArrayList<>(pointList);
                vertexes.remove(2); // 第3个关键点在车牌中无意义
                // 排序后判断坐标方位
                vertexes.sort((a, b) -> Float.compare(a.x, b.x));
                result.plateVertexes.lt = vertexes.get(0).y < vertexes.get(1).y ? vertexes.get(0) : vertexes.get(1);
                result.plateVertexes.rt = vertexes.get(2).y < vertexes.get(3).y ? vertexes.get(2) : vertexes.get(3);
                result.plateVertexes.rb = vertexes.get(2).y >= vertexes.get(3).y ? vertexes.get(2) : vertexes.get(3);
                result.plateVertexes.lb = vertexes.get(0).y >= vertexes.get(1).y ? vertexes.get(0) : vertexes.get(1);
            } else if (result.classId == 1) {
                result.faceLandmarks = new Face.Landmarks(pointList.get(0), pointList.get(1), pointList.get(2), pointList.get(3), pointList.get(4));
                result.faceAngles = calculateAngles(result.faceLandmarks);
            }
            results.add(result);
        }
        // NMS
        results.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        Set<Integer> nmsIndexSet = new HashSet<>();
        for (int i = 0; i < results.size() - 1; i++) {
            Box box1 = results.get(i).box;
            for (int j = i + 1; j < results.size(); j++) {
                Box box2 = results.get(j).box;
                if (iou(box1, box2) >= IOU_THRESHOLD) {
                    nmsIndexSet.add(j);
                }
            }
        }
        List<Integer> nmsIndexList = new ArrayList<>(nmsIndexSet);
        nmsIndexList.sort((a, b) -> Integer.compare(b, a)); // 降序排列
        for (int index : nmsIndexList) {
            results.remove(index);
        }
        return results;
    }

    // 计算交并比
    private static float iou(Box box1, Box box2) {
        int width = Math.max(0, box1.point.x < box2.point.x ? (box1.point.x + box1.width - box2.point.x) : (box2.point.x + box2.width - box1.point.x));
        int height = Math.max(0, box1.point.y < box2.point.y ? (box1.point.y + box1.height - box2.point.y) : (box2.point.y + box2.height - box1.point.y));
        int area1 = box1.width * box1.height;
        int area2 = box2.width * box2.height;
        int area = width * height;
        int union = area1 + area2 - area;
        if (union <= 0) {
            return 0.0f;
        }
        return (float) area / union;
    }

    // 计算姿态
    private static Face.Angles calculateAngles(Face.Landmarks landmarks) {
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

}
