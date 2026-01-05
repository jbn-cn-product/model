package com.example.model;

import com.example.model.base.OnnxDeployer;
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
        public int[] bbox;              // 检测框
        public float confidence;        // 置信度
        public int classId;             // 类别 0-车牌 1-人脸
        // 以下成员在classId为人脸时才有意义
        public List<float[]> landmarks; // 关键点
        public float yaw;               // 偏航角
        public float pitch;             // 俯仰角
        public float roll;              // 翻滚角
    }

    private static final String TAG = "MyLogcat-FacePlateDetector";

    // 模型参数
    public static final int MODEL_WIDTH = 640;
    public static final int MODEL_HEIGHT = 640;
    private static final float CONF_THRESHOLD = 0.3f;   // 置信度阈值
    private static final float IOU_THRESHOLD = 0.5f;    // 重合阈值
    private static final float MEAN_VALUE = 0.0f;
    private static final float STD_VALUE = 1.5f;

    // 输入图像尺寸
    private int inputWidth;
    private int inputHeight;

    public FacePlateDetector(byte[] modelData, Logger logger) {
        super(modelData, logger, MODEL_WIDTH, MODEL_HEIGHT, MEAN_VALUE, STD_VALUE);
    }

    // 运行
    public List<Result> run(byte[] rgbData, int inputWidth, int inputHeight) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
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
            logger.error(TAG, "update output tensor failed");
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
        List<float[]> bboxList = new ArrayList<>();
        List<Float> thresholdList = new ArrayList<>();
        List<Integer> classIdList = new ArrayList<>();
        List<float[]> landmarksList = new ArrayList<>();
        float scale = Math.min((float) MODEL_WIDTH / inputWidth, (float) MODEL_HEIGHT / inputHeight);
        int padX = (MODEL_WIDTH - (int)(inputWidth * scale)) / 2;
        int padY = (MODEL_HEIGHT - (int)(inputHeight * scale)) / 2;
        // 解析输出结构
        for (float[] output : outputs) {
            float confidence = output[4];
            float plateScore = output[15];
            float faceScore = output[16];
            float finalConfidence = confidence * Math.max(plateScore, faceScore); // 输出置信度*最高类别分数=最终置信度
            if (finalConfidence < CONF_THRESHOLD) {
                continue;
            }
            thresholdList.add(finalConfidence);
            classIdList.add(plateScore > faceScore ? 0 : 1);
            // 输出的检测框和特征点坐标缩放到原图比例
            float x = output[0], y = output[1], w = output[2], h = output[3];
            float[] bbox = new float[4];
            bbox[0] = Math.max(0, Math.min(inputWidth, (x - w / 2 - padX) / scale));
            bbox[1] = Math.max(0, Math.min(inputHeight, (y - h / 2 - padY) / scale));
            bbox[2] = Math.max(0, Math.min(inputWidth, (x + w / 2 - padX) / scale));
            bbox[3] = Math.max(0, Math.min(inputHeight, (y + h / 2 - padY) / scale));
            bboxList.add(bbox);
            float[] landmarks = new float[10];
            for (int i = 0; i < 10; i += 2) {
                landmarks[i] = Math.max(0, Math.min(inputWidth, (output[5 + i] - padX) / scale));
                landmarks[i + 1] = Math.max(0, Math.min(inputHeight, (output[6 + i] - padY) / scale));
            }
            landmarksList.add(landmarks);
        }
        // NMS过滤多个重复检测框
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < thresholdList.size(); i++) {
            order.add(i);
        }
        order.sort((a, b) -> Float.compare(thresholdList.get(b), thresholdList.get(a)));
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
        for (int keepIndex : keepIndexes) {
            float[] bbox = bboxList.get(keepIndex);
            Result result = new Result();
            result.bbox = new int[]{(int) bbox[0], (int) bbox[1], (int) bbox[2], (int) bbox[3]};
            result.confidence = thresholdList.get(keepIndex);
            result.classId = classIdList.get(keepIndex);
            if (result.classId == 1) {
                float[] landmarks = landmarksList.get(keepIndex);
                List<float[]> landmarkPoints = new ArrayList<>();
                for (int i = 0; i < 5; i++) {
                    landmarkPoints.add(new float[]{landmarks[i * 2], landmarks[i * 2 + 1]});
                }
                result.landmarks = landmarkPoints;
                float[] leftEye = result.landmarks.get(0);
                float[] rightEye = result.landmarks.get(1);
                float[] nose = result.landmarks.get(2);
                float[] mouthLeft = result.landmarks.get(3);
                float[] mouthRight = result.landmarks.get(4);
                float eyeCenterY = (leftEye[1] + rightEye[1]) / 2.0f;
                float mouthCenterY = (mouthLeft[1] + mouthRight[1]) / 2.0f;
                float expectedNoseY = (eyeCenterY + mouthCenterY) / 2.0f;
                float noseDeviation = nose[1] - expectedNoseY;
                float faceHeight = Math.abs(mouthCenterY - eyeCenterY);
                float pitch;
                if (faceHeight > 0.0f) {
                    float ratio = noseDeviation / faceHeight;
                    if (ratio < -1.0f) ratio = -1.0f;
                    if (ratio > 1.0f) ratio = 1.0f;
                    pitch = ratio * 20.0f;
                } else {
                    pitch = 0.0f;
                }
                float eyeCenterX = (leftEye[0] + rightEye[0]) / 2.0f;
                float eyeDistance = Math.abs(rightEye[0] - leftEye[0]);
                float noseOffset = nose[0] - eyeCenterX;
                float yaw;
                if (eyeDistance > 0.0f) {
                    float ratio = noseOffset / eyeDistance;
                    if (ratio < -1.0f) ratio = -1.0f;
                    if (ratio > 1.0f) ratio = 1.0f;
                    yaw = ratio * 30.0f;
                } else {
                    yaw = 0.0f;
                }
                float eyeDx = rightEye[0] - leftEye[0];
                float eyeDy = rightEye[1] - leftEye[1];
                float roll;
                if (Math.abs(eyeDx) > 0.0f) {
                    roll = (float) Math.toDegrees(Math.atan(eyeDy / eyeDx));
                } else {
                    roll = 0.0f;
                }
                result.pitch = pitch;
                result.yaw = yaw;
                result.roll = roll;
            }
            results.add(result);
        }
        return results;
    }

}
